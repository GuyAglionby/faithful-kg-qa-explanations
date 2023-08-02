import logging
import math

import torch.nn
from torch import nn

from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from torch.autograd import Variable
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter

from utils.data_utils import MultiGPUSparseAdjDataBatchGenerator, load_sparse_adj_data_with_contextnode, \
    load_input_tensors
from utils.layers import GELU, MultiheadAttPoolLayer, MLP, CustomizedEmbedding

logger = logging.getLogger(__name__)


class QAGNN_Message_Passing(nn.Module):

    def __init__(self, k, n_ntype, n_etype,
                 input_size, hidden_size, output_size,
                 dropout=0.1, ablation=None):
        super().__init__()
        assert input_size == output_size
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.ablation = ablation

        self.basis_f = 'sin'  # ['id', 'linact', 'sin', 'none']
        if 'no_node_scoring' not in self.ablation:
            self.emb_node_type = nn.Linear(self.n_ntype, hidden_size // 2)

            if self.basis_f in ['id']:
                self.emb_score = nn.Linear(1, hidden_size // 2)
            elif self.basis_f in ['linact']:
                self.B_lin = nn.Linear(1, hidden_size // 2)
                self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)
            elif self.basis_f in ['sin']:
                self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)
        else:
            # This is what means we don't have to make a whole load of changes elsewhere
            # (which assume a vector of hidden_size, which we would otherwise get by catting node type vec w/ score vec)
            self.emb_node_type = nn.Linear(self.n_ntype, hidden_size)
            self.emb_score = None

        use_batchnorm = 'no_batchnorm' not in self.ablation
        batchnorm_layer = torch.nn.BatchNorm1d if use_batchnorm else torch.nn.Identity
        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size),
                                                batchnorm_layer(hidden_size),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_size, hidden_size))

        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(hidden_size, n_ntype, n_etype,
                                                  self.edge_encoder, use_batchnorm=use_batchnorm) for _ in range(k)])

        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra):
        for i in range(self.k):
            _X = self.gnn_layers[i](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training=self.training)
        return _X

    def forward(self, H, A, node_type, node_score):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        # Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T))  # [batch_size, n_node, dim/2] OR dim if no_node_scoring

        # Embed score
        if 'no_node_scoring' not in self.ablation:
            if self.basis_f == 'sin':
                # [1,1,dim/2]
                js = torch.arange(self.hidden_size // 2).unsqueeze(0).unsqueeze(0).float().to(node_type.device)
                # [1,1,dim/2]
                js = torch.pow(1.1, js)
                # [batch_size, n_node, dim/2]
                B = torch.sin(js * node_score)
                # [batch_size, n_node, dim/2]
                node_score_emb = self.activation(self.emb_score(B))
            elif self.basis_f == 'id':
                B = node_score
                # [batch_size, n_node, dim/2]
                node_score_emb = self.activation(self.emb_score(B))
            elif self.basis_f == 'linact':
                # [batch_size, n_node, dim/2]
                B = self.activation(self.B_lin(node_score))
                # [batch_size, n_node, dim/2]
                node_score_emb = self.activation(self.emb_score(B))
            else:
                raise ValueError()
        else:
            node_score_emb = None

        X = H
        # edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        edge_index, edge_type = A
        # [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _X = X.view(-1, X.size(2)).contiguous()
        # [`total_n_nodes`, ]
        _node_type = node_type.view(-1).contiguous()
        # [`total_n_nodes`, dim]
        if 'no_node_scoring' not in self.ablation:
            # [batch_size * n_node, dim]
            _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0),
                                                                                         -1).contiguous()
        else:
            # [batch_size * n_node, dim] (STILL dim NOT dim/2 because of how we size at init)
            _node_feature_extra = node_type_emb.view(_node_type.size(0), -1).contiguous()

        _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)

        # [batch_size, n_node, dim]
        X = _X.view(node_type.size(0), node_type.size(1), -1)

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return output


class QAGNN(nn.Module):
    def __init__(self, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, ablation=None):
        super().__init__()
        self.init_range = init_range
        self.ablation = ablation if ablation is not None else {}

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=False, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb,
                                               freeze_ent_emb=freeze_ent_emb)
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)

        self.concept_dim = concept_dim

        self.activation = GELU()

        self.gnn = QAGNN_Message_Passing(k=k, n_ntype=n_ntype, n_etype=n_etype,
                                         input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim,
                                         dropout=p_gnn, ablation=ablation)

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        fc_input_dim = concept_dim
        if 'no_s_in_final_mlp_direct' not in self.ablation:
            fc_input_dim += sent_dim
        if 'no_s_in_final_mlp_from_graph' not in self.ablation:
            fc_input_dim += concept_dim

        self.fc = MLP(fc_input_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None,
                cache_output=False):
        """
        sent_vecs: (batch_size, dim_sent)
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """
        # (batch_size, 1, dim_node)
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1)
        # (batch_size, n_node-1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:] - 1, emb_data).to(node_type_ids.device)
        # (batch_size, n_node, dim_node)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1))

        # Normalize node sore (use norm from Z)
        # 0 means masked out #[batch_size, n_node]
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float()
        node_scores = -node_scores
        # [batch_size, n_node, 1]
        node_scores = node_scores - node_scores[:, 0:1, :]
        # [batch_size, n_node]
        node_scores = node_scores.squeeze(2)
        node_scores = node_scores * _mask
        # [batch_size, ]
        mean_norm = (torch.abs(node_scores)).sum(dim=1) / adj_lengths
        # [batch_size, n_node]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05)
        # [batch_size, n_node, 1]
        node_scores = node_scores.unsqueeze(2)

        gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores)

        # (batch_size, dim_node)
        Z_vecs = gnn_output[:, 0]

        # 1 means masked out
        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1)

        mask = mask | (node_type_ids == 3)  # pool over all KG nodes
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        # No need to do ablation here, mask on the ctx node already exists
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)
        # if not 'no_s_in_final_mlp_from_graph_pool' in self.ablation:
        #     graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)
        # else:
        #     graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output[:, 1:], mask)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        vecs_to_cat = [graph_vecs]
        if 'no_s_in_final_mlp_direct' not in self.ablation:
            vecs_to_cat.append(sent_vecs)
        if 'no_s_in_final_mlp_from_graph' not in self.ablation:
            vecs_to_cat.append(Z_vecs)
        catted = torch.cat(vecs_to_cat, 1)
        logits = self.fc(self.dropout_fc(catted))

        return logits, (pool_attn.cpu(), mask.cpu())


class LM_QAGNN(nn.Module):
    def __init__(self, model_name, k, n_ntype, n_etype,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config=None, ablation=None):
        super().__init__()
        if encoder_config is None:
            encoder_config = {}
        self.ablation = ablation if ablation is not None else {}
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = QAGNN(k, n_ntype, n_etype, self.encoder.sent_dim,
                             n_concept, concept_dim, concept_in_dim, n_attention_head,
                             fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                             pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                             init_range=init_range, ablation=self.ablation)

    def forward(self, *inputs, layer_id=-1, cache_output=False, detail=False):
        """
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        adj_lengths: (batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )
        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        # Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = ([x.view(x.size(0) * x.size(1), *x.size()[2:])
                    for x in inputs[:-6]] +
                   [x.view(x.size(0) * x.size(1), *x.size()[2:])
                    for x in inputs[-6:-2]] +
                   [sum(x, [])
                    for x in inputs[-2:]])

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        # edge_index: [2, total_E]   edge_type: [total_E, ]
        adj = (edge_index.to(node_type_ids.device),
               edge_type.to(node_type_ids.device))

        sent_vecs, _ = self.encoder(*lm_inputs, layer_id=layer_id)
        if 'detach_s_all' in self.ablation:
            sent_vecs = sent_vecs.detach()
        logits, attn = self.decoder(sent_vecs.to(node_type_ids.device),
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj,
                                    emb_data=None, cache_output=cache_output)
        logits = logits.view(bs, nc)
        if not detail:
            return logits, attn
        else:
            return logits, attn, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc,
                                                                                  -1), edge_index_orig, edge_type_orig
            # edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            # edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )

    @staticmethod
    def batch_graph(edge_index_init, edge_type_init, n_nodes):
        # edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        # edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E, ]
        return edge_index, edge_type


class LM_QAGNN_DataLoader(object):

    def __init__(self, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        logger.info('train_statement_path %s', train_statement_path)
        self.train_qids, self.train_labels, self.train_n_labs, *self.train_encoder_data = load_input_tensors(
            train_statement_path, model_type, model_name, max_seq_length, ret_num_labs=True)
        self.dev_qids, self.dev_labels, self.dev_n_labs, *self.dev_encoder_data = load_input_tensors(dev_statement_path,
                                                                                                     model_type,
                                                                                                     model_name,
                                                                                                     max_seq_length,
                                                                                                     ret_num_labs=True)

        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        logger.info('num_choice %d', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path,
                                                                                              max_node_num, num_choice)

        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num,
                                                                                          num_choice)
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in
                   [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in
                   [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, self.test_n_labs_, *self.test_encoder_data = load_input_tensors(
                test_statement_path, model_type, model_name, max_seq_length, ret_num_labs=True)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path,
                                                                                                max_node_num,
                                                                                                num_choice)
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in
                       [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids)
                                                       if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids)
                                                      if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0)
                           for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    @property
    def test_n_labs(self):
        if self.is_inhouse:
            return {self.train_qids[idx]: self.train_n_labs[self.train_qids[idx]] for idx in self.inhouse_test_indexes}
        elif hasattr(self, 'test_n_labs_'):
            return self.test_n_labs_
        else:
            return {}

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.batch_size, train_indexes,
                                                   self.train_qids, self.train_labels, tensors0=self.train_encoder_data,
                                                   tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size,
                                                   torch.arange(len(self.train_qids)), self.train_qids,
                                                   self.train_labels, tensors0=self.train_encoder_data,
                                                   tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size,
                                                   torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels,
                                                   tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data,
                                                   adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size,
                                                       self.inhouse_test_indexes, self.train_qids, self.train_labels,
                                                       tensors0=self.train_encoder_data,
                                                       tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size,
                                                       torch.arange(len(self.test_qids)), self.test_qids,
                                                       self.test_labels, tensors0=self.test_encoder_data,
                                                       tensors1=self.test_decoder_data, adj_data=self.test_adj_data)


###############################################################################
############################### GNN architecture ##############################
###############################################################################

def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """

    def __init__(self, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add", use_batchnorm=True):
        super(GATConvE, self).__init__(aggr=aggr)
        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype
        self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        # For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2 * emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        batchnorm_layer = torch.nn.BatchNorm1d if use_batchnorm else torch.nn.Identity
        # For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),
                                       batchnorm_layer(emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=False):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]

        # Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype + 1)  # [E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype + 1).to(edge_vec.device)
        self_edge_vec[:, self.n_etype] = 1

        head_type = node_type[edge_index[0]]  # [E,] #head=src
        tail_type = node_type[edge_index[1]]  # [E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype)  # [N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)  # [N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)  # [E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]

        # Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim]
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2 * self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        # key from target
        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim]
        # query from source
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        out = out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]

        return out
