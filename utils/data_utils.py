import itertools
import logging
import pickle
import os

# import dgl
import numpy as np
import torch
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, AlbertTokenizer,
                          AutoTokenizer)

import json
from tqdm import tqdm

# from utils.tokenization_utils import WordTokenizer, PAD_TOK

GPT_SPECIAL_TOKENS = ['_start_', '_delimiter_', '_classify_']

logger = logging.getLogger(__name__)


class BatchGenerator(object):
    def __init__(self, device, batch_size, indexes, qids, labels, tensors=[], lists=[]):
        self.device = device
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors = tensors
        self.lists = lists

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes])
            batch_tensors = [self._to_device(x[batch_indexes]) for x in self.tensors]
            batch_lists = [self._to_device([x[i] for i in batch_indexes]) for x in self.lists]
            yield tuple([batch_qids, batch_labels, *batch_tensors, *batch_lists])

    def _to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item) for item in obj]
        else:
            return obj.to(self.device)


# class MultiGPUBatchGenerator(object):
#     def __init__(self, device0, device1, batch_size, indexes, qids, labels, tensors0=[], lists0=[], tensors1=[], lists1=[]):
#         self.device0 = device0
#         self.device1 = device1
#         self.batch_size = batch_size
#         self.indexes = indexes
#         self.qids = qids
#         self.labels = labels
#         self.tensors0 = tensors0
#         self.lists0 = lists0
#         self.tensors1 = tensors1
#         self.lists1 = lists1
#
#     def __len__(self):
#         return (self.indexes.size(0) - 1) // self.batch_size + 1
#
#     def __iter__(self):
#         bs = self.batch_size
#         n = self.indexes.size(0)
#         for a in range(0, n, bs):
#             b = min(n, a + bs)
#             batch_indexes = self.indexes[a:b]
#             batch_qids = [self.qids[idx] for idx in batch_indexes]
#             batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
#             batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
#             batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
#             batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
#             batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]
#
#             yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1])
#
#     def _to_device(self, obj, device):
#         if isinstance(obj, (tuple, list)):
#             return [self._to_device(item, device) for item in obj]
#         else:
#             return obj.to(device)


# class AdjDataBatchGenerator(object):
#     def __init__(self, device, batch_size, indexes, qids, labels, tensors=[], lists=[], adj_empty=None, adj_data=None):
#         self.device = device
#         self.batch_size = batch_size
#         self.indexes = indexes
#         self.qids = qids
#         self.labels = labels
#         self.tensors = tensors
#         self.lists = lists
#         self.adj_empty = adj_empty
#         self.adj_data = adj_data
#
#     def __len__(self):
#         return (self.indexes.size(0) - 1) // self.batch_size + 1
#
#     def __iter__(self):
#         batch_adj = self.adj_empty  # (batch_size, num_choice, n_rel, n_node, n_node)
#         batch_adj[:] = 0
#         batch_adj[:, :, -1] = torch.eye(batch_adj.size(-1), dtype=torch.float32, device=self.device)
#         bs = self.batch_size
#         n = self.indexes.size(0)
#         for a in range(0, n, bs):
#             b = min(n, a + bs)
#             batch_indexes = self.indexes[a:b]
#             batch_qids = [self.qids[idx] for idx in batch_indexes]
#             batch_labels = self._to_device(self.labels[batch_indexes])
#             batch_tensors = [self._to_device(x[batch_indexes]) for x in self.tensors]
#             batch_lists = [self._to_device([x[i] for i in batch_indexes]) for x in self.lists]
#
#             batch_adj[:, :, :-1] = 0
#             for batch_id, global_id in enumerate(batch_indexes):
#                 for choice_id, (i, j, k) in enumerate(self.adj_data[global_id]):
#                     batch_adj[batch_id, choice_id, i, j, k] = 1
#
#             yield tuple([batch_qids, batch_labels, *batch_tensors, *batch_lists, batch_adj[:b - a]])
#
#     def _to_device(self, obj):
#         if isinstance(obj, (tuple, list)):
#             return [self._to_device(item) for item in obj]
#         else:
#             return obj.to(self.device)


class MultiGPUAdjDataBatchGenerator(object):
    """
    this version DOES NOT add the identity matrix
    tensors0, lists0  are on device0
    tensors1, lists1, adj, labels  are on device1
    """

    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_empty=None, adj_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        # need this to avoid memory usage exploding
        torch.cuda.empty_cache()
        self.adj_empty = adj_empty.to(self.device1)
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        batch_adj = self.adj_empty  # (batch_size, num_choice, n_rel, n_node, n_node)
        batch_adj[:] = 0
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            batch_adj[:] = 0
            for batch_id, global_id in enumerate(batch_indexes):
                for choice_id, (i, j, k) in enumerate(self.adj_data[global_id]):
                    batch_adj[batch_id, choice_id, i, j, k] = 1

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, batch_adj[:b - a]])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


# class MultiGPUNxgDataBatchGenerator(object):
#     """
#     tensors0, lists0  are on device0
#     tensors1, lists1, adj, labels  are on device1
#     """
#
#     def __init__(self, device0, device1, batch_size, indexes, qids, labels,
#                  tensors0=[], lists0=[], tensors1=[], lists1=[], graph_data=None):
#         self.device0 = device0
#         self.device1 = device1
#         self.batch_size = batch_size
#         self.indexes = indexes
#         self.qids = qids
#         self.labels = labels
#         self.tensors0 = tensors0
#         self.lists0 = lists0
#         self.tensors1 = tensors1
#         self.lists1 = lists1
#         self.graph_data = graph_data
#
#     def __len__(self):
#         return (self.indexes.size(0) - 1) // self.batch_size + 1
#
#     def __iter__(self):
#         bs = self.batch_size
#         n = self.indexes.size(0)
#         for a in range(0, n, bs):
#             b = min(n, a + bs)
#             batch_indexes = self.indexes[a:b]
#             batch_qids = [self.qids[idx] for idx in batch_indexes]
#             batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
#             batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
#             batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
#             batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
#             # qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data
#             batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]
#
#             # (batch, n_answer_cands)
#             this_batch_graph_data = [self.graph_data[instance_index] for instance_index in batch_indexes]
#             # flat_graph_data = sum(this_batch_graph_data, [])
#             concept_mapping_dicts = []
#             acc_start = 0
#
#             # print("batch_lists1", len(batch_lists1[1]), len(batch_lists1[1][0]), type(batch_lists1[1][0]))
#             # print("flat_graph_data", len(flat_graph_data), len(flat_graph_data[0]), type(flat_graph_data[0]))
#
#             # qa_pairs = batch_lists1[0]
#             # print(qa_pairs, len(qa_pairs), len(qa_pairs[0]))
#             # qa_pairs_max =
#
#             for batched_graph_data, batched_cpt_paths, batched_qa_pairs in zip(this_batch_graph_data, batch_lists1[1], batch_lists1[0]):
#                 # Length can be 0 for padding elements
#                 qa_pairs_max = max([batch_item.max().item() if batch_item.shape[0] > 0 else 0 for batch_item in batched_qa_pairs])
#                 for g, cpt_path in zip(batched_graph_data, batched_cpt_paths):
#                     # concept_mapping_dict = {}
#                     if len(cpt_path) == 0:
#                         cpt_path_max = qa_pairs_max
#                     else:
#                         cpt_path_max = max(cpt_path.max().item(), qa_pairs_max)
#                     vec = torch.full((cpt_path_max + 1,), -9999).to(cpt_path.device)
#                     for index, cncpt_id in enumerate(g.ndata['cncpt_ids']):
#                         if int(cncpt_id) > cpt_path_max:
#                             continue
#                         vec[int(cncpt_id)] = acc_start + index
#                         # concept_mapping_dict[int(cncpt_id)] = acc_start + index
#                         # concept_mapping_dict_v[int(cncpt_id)] = acc_start + index
#                     acc_start += len(g.nodes())
#                     concept_mapping_dicts.append(vec)
#                 # concept_mapping_dicts.append(concept_mapping_dict_v)
#             batched_graph = dgl.batch(list(itertools.chain.from_iterable(this_batch_graph_data))).to(self.device1)
#             # batched_graph.ndata['cncpt_ids'] = batched_graph.ndata['cncpt_ids'].to(self.device1)
#
#             yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_tensors1, *batch_lists0, *batch_lists1, batched_graph, concept_mapping_dicts])
#
#     def _to_device(self, obj, device):
#         if isinstance(obj, (tuple, list)):
#             return [self._to_device(item, device) for item in obj]
#         else:
#             return obj.to(device)


def load_sparse_adj_data_with_contextnode(adj_pk_path, max_node_num, num_choice):
    cache_path = adj_pk_path + '.loaded_cache'
    use_cache = True

    if use_cache and not os.path.exists(cache_path):
        use_cache = False

    if use_cache:
        with open(cache_path, 'rb') as f:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel = pickle.load(f)
    else:
        with open(adj_pk_path, 'rb') as fin:
            adj_concept_pairs = pickle.load(fin)

        n_samples = len(adj_concept_pairs) # this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long) # default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)

        adj_lengths_ori = adj_lengths.clone()
        for idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
            adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score']
            # adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            # concepts: np.array(num_nodes, ), where entry is concept id
            # qm: np.array(num_nodes, ), where entry is True/False
            # am: np.array(num_nodes, ), where entry is True/False
            assert len(concepts) == len(set(concepts))
            qam = qm | am
            # sanity check: should be T,..,T,F,F,..F
            assert qam[0]
            F_start = False
            for TF in qam:
                # ga384 the `not` here is new. possibility that it introduces bugs
                if not TF:
                    F_start = True
                else:
                    assert not F_start
            num_concept = min(len(concepts), max_node_num - 1) + 1 # this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            # Prepare nodes
            concepts = concepts[:num_concept-1]
            concept_ids[idx, 1:num_concept] = torch.tensor(concepts + 1)  # To accomodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = 0 # this is the "concept_id" for contextnode

            # Prepare node scores
            if cid2score is not None:
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1
                    assert _cid in cid2score
                node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            # Prepare node types
            node_type_ids[idx, 0] = 3 # contextnode
            node_type_ids[idx, 1:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept-1]] = 0
            node_type_ids[idx, 1:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept-1]] = 1

            # Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            # Prepare edges
            i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
            extra_i, extra_j, extra_k = [], [], []
            for _coord, q_tf in enumerate(qm):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if q_tf:
                    extra_i.append(0) # rel from contextnode to question concept
                    extra_j.append(0) # contextnode coordinate
                    extra_k.append(_new_coord) # question concept coordinate
            for _coord, a_tf in enumerate(am):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if a_tf:
                    extra_i.append(1) # rel from contextnode to answer concept
                    extra_j.append(0) # contextnode coordinate
                    extra_k.append(_new_coord) # answer concept coordinate

            half_n_rel += 2 # should be 19 now
            if len(extra_i) > 0:
                i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                k = torch.cat([k, torch.tensor(extra_k)], dim=0)
            ########################

            mask = (j < max_node_num) & (k < max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
            edge_index.append(torch.stack([j,k], dim=0)) # each entry is [2, E]
            edge_type.append(i) # each entry is [E, ]

        with open(cache_path, 'wb') as f:
            pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel], f)

    ori_adj_mean  = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
    logger.info('| ori_adj_len: mu %.2f sigma %.2f | adj_len: %.2f | prune_rate：%.2f | qc_num: %.2f | ac_num: %.2f |',
                ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item(),
                (adj_lengths_ori > adj_lengths).float().mean().item(),
                (node_type_ids == 0).float().sum(1).mean().item(),
                (node_type_ids == 1).float().sum(1).mean().item())

    edge_index = list(map(list, zip(*(iter(edge_index),) * num_choice))) # list of size (n_questions, n_choices), where each entry is tensor[2, E] # this operation corresponds to .view(n_questions, n_choices)
    edge_type = list(map(list, zip(*(iter(edge_type),) * num_choice))) # list of size (n_questions, n_choices), where each entry is tensor[E, ]

    concept_ids, node_type_ids, node_scores, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths)]
    # concept_ids: (n_questions, num_choice, max_node_num)
    # node_type_ids: (n_questions, num_choice, max_node_num)
    # node_scores: (n_questions, num_choice, max_node_num)
    # adj_lengths: (n_questions,　num_choice)
    return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type) #, half_n_rel * 2 + 1


class MultiGPUSparseAdjDataBatchGenerator(object):
    def __init__(self, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None):
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        # self.adj_empty = adj_empty.to(self.device1)
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]


            edge_index_all, edge_type_all = self.adj_data
            # edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            # edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type  = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, edge_index, edge_type])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


# def load_2hop_relational_paths(rpath_jsonl_path, cpt_jsonl_path=None, emb_pk_path=None,
#                                max_tuple_num=200, num_choice=None, node_feature_type=None):
#     with open(rpath_jsonl_path, 'r') as fin:
#         rpath_data = [json.loads(line) for line in fin]
#
#     with open(cpt_jsonl_path, 'rb') as fin:
#         adj_data = pickle.load(fin)  # (adj, concepts, qm, am)
#
#     n_samples = len(rpath_data)
#     qa_data = torch.zeros((n_samples, max_tuple_num, 2), dtype=torch.long)
#     rel_data = torch.zeros((n_samples, max_tuple_num), dtype=torch.long)
#     num_tuples = torch.zeros((n_samples,), dtype=torch.long)
#
#     all_masks = []
#     for i, (data, adj) in enumerate(tqdm(zip(rpath_data, adj_data), total=n_samples, desc='loading QA pairs')):
#         concept_ids = adj[1]
#         ori_cpt2idx = {c: i for (i, c) in enumerate(concept_ids)}
#         qa_mask = np.zeros(len(concept_ids), dtype=np.bool)
#
#         cur_qa = []
#         cur_rel = []
#         for dic in data['paths']:
#             if len(dic['rel']) == 1:
#                 cur_qa.append([dic['qc'], dic['ac']])
#                 cur_rel.append(dic['rel'][0])
#             elif len(dic['rel']) == 2:
#                 cur_qa.append([dic['qc'], dic['ac']])
#                 cur_rel.append(34 + dic['rel'][0] * 34 + dic['rel'][1])
#             else:
#                 raise ValueError('Invalid path length')
#             qa_mask[ori_cpt2idx[dic['qc']]] = True
#             qa_mask[ori_cpt2idx[dic['ac']]] = True
#             if len(cur_qa) >= max_tuple_num:
#                 break
#         assert len(cur_qa) == len(cur_rel)
#         all_masks.append(qa_mask)
#
#         if len(cur_qa) > 0:
#             qa_data[i][:len(cur_qa)] = torch.tensor(cur_qa)
#             rel_data[i][:len(cur_rel)] = torch.tensor(cur_rel)
#             num_tuples[i] = (len(cur_qa) + len(cur_rel)) // 2  # code style suggested by kiwisher
#
#     if emb_pk_path is not None:  # use contexualized node features
#         with open(emb_pk_path, 'rb') as fin:
#             all_embs = pickle.load(fin)
#         assert len(all_embs) == len(all_masks) == n_samples
#         max_cpt_num = max(mask.sum() for mask in all_masks)
#         if node_feature_type in ('cls', 'mention'):
#             emb_dim = all_embs[0].shape[1] // 2
#         else:
#             emb_dim = all_embs[0].shape[1]
#         emb_data = torch.zeros((n_samples, max_cpt_num, emb_dim), dtype=torch.float)
#         for idx, (mask, embs) in enumerate(zip(all_masks, all_embs)):
#             assert not any(mask[embs.shape[0]:])
#             masked_concept_ids = adj_data[idx][1][mask]
#             masked_embs = embs[mask[:embs.shape[0]]]
#             cpt2idx = {c: i for (i, c) in enumerate(masked_concept_ids)}
#             for tuple_idx in range(num_tuples[idx].item()):
#                 qa_data[idx, tuple_idx, 0] = cpt2idx[qa_data[idx, tuple_idx, 0].item()]
#                 qa_data[idx, tuple_idx, 1] = cpt2idx[qa_data[idx, tuple_idx, 1].item()]
#             if node_feature_type in ('cls',):
#                 masked_embs = masked_embs[:, :emb_dim]
#             elif node_feature_type in ('mention',):
#                 masked_embs = masked_embs[:, emb_dim:]
#             emb_data[idx, :masked_embs.shape[0]] = torch.tensor(masked_embs)
#             assert (qa_data[idx, :num_tuples[idx]] < masked_embs.shape[0]).all()
#
#     if num_choice is not None:
#         qa_data = qa_data.view(-1, num_choice, max_tuple_num, 2)
#         rel_data = rel_data.view(-1, num_choice, max_tuple_num)
#         num_tuples = num_tuples.view(-1, num_choice)
#         if emb_pk_path is not None:
#             emb_data = emb_data.view(-1, num_choice, *emb_data.size()[1:])
#
#     flat_rel_data = rel_data.view(-1, max_tuple_num)
#     flat_num_tuples = num_tuples.view(-1)
#     valid_mask = (torch.arange(max_tuple_num) < flat_num_tuples.unsqueeze(-1)).float()
#     n_1hop_paths = ((flat_rel_data < 34).float() * valid_mask).sum(1)
#     n_2hop_paths = ((flat_rel_data >= 34).float() * valid_mask).sum(1)
#     logger.info("| #paths %f | average #1-hop paths: %f | average #2-hop paths: %f | #w/ 1-hop %f | #w/ 2-hop %f |",
#                 flat_num_tuples.float().mean(0),
#                 n_1hop_paths.mean(), n_2hop_paths.mean(),
#                 (n_1hop_paths > 0).float().mean(), (n_2hop_paths > 0).float().mean())
#     return (qa_data, rel_data, num_tuples, emb_data) if emb_pk_path is not None else (qa_data, rel_data, num_tuples)


def load_adj_data(adj_pk_path, max_node_num, num_choice, emb_pk_path=None):
    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)

    n_samples = len(adj_concept_pairs)
    adj_data = []
    adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
    concept_ids = torch.zeros((n_samples, max_node_num), dtype=torch.long)
    node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)

    if emb_pk_path is not None:
        with open(emb_pk_path, 'rb') as fin:
            all_embs = pickle.load(fin)
        emb_data = torch.zeros((n_samples, max_node_num, all_embs[0].shape[1]), dtype=torch.float)
    else:
        # make pycharm happy
        emb_data = None

    adj_lengths_ori = adj_lengths.clone()
    for idx, adj_data_instance in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
        if isinstance(adj_data_instance, tuple):
            adj, concepts, qm, am = adj_data_instance
        elif isinstance(adj_data_instance, dict):
            adj = adj_data_instance['adj']
            concepts = adj_data_instance['concepts']
            qm = adj_data_instance['qmask']
            am = adj_data_instance['amask']
        else:
            raise ValueError()
        num_concept = min(len(concepts), max_node_num)
        adj_lengths_ori[idx] = len(concepts)
        if emb_pk_path is not None:
            embs = all_embs[idx]
            assert embs.shape[0] >= num_concept
            emb_data[idx, :num_concept] = torch.tensor(embs[:num_concept])
            concepts = np.arange(num_concept)
        else:
            concepts = concepts[:num_concept]
        concept_ids[idx, :num_concept] = torch.tensor(concepts)  # note : concept zero padding is disabled

        adj_lengths[idx] = num_concept
        node_type_ids[idx, :num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept]] = 0
        node_type_ids[idx, :num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept]] = 1
        ij = torch.tensor(adj.row, dtype=torch.int64)
        k = torch.tensor(adj.col, dtype=torch.int64)
        n_node = adj.shape[1]
        half_n_rel = adj.shape[0] // n_node
        i, j = ij // n_node, ij % n_node
        mask = (j < max_node_num) & (k < max_node_num)
        i, j, k = i[mask], j[mask], k[mask]
        i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
        adj_data.append((i, j, k))  # i, j, k are the coordinates of adj's non-zero entries

    logger.info("| ori_adj_len: %.2f | adj_len: %.2f | prune_rate： %.2f | qc_num: %.2f | ac_num: %.2f |",
                adj_lengths_ori.float().mean().item(),
                adj_lengths.float().mean().item(),
                (adj_lengths_ori > adj_lengths).float().mean().item(),
                (node_type_ids == 0).float().sum(1).mean().item(),
                (node_type_ids == 1).float().sum(1).mean().item())

    concept_ids, node_type_ids, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, adj_lengths)]
    if emb_pk_path is not None:
        emb_data = emb_data.view(-1, num_choice, *emb_data.size()[1:])
    adj_data = list(map(list, zip(*(iter(adj_data),) * num_choice)))

    if emb_pk_path is None:
        return concept_ids, node_type_ids, adj_lengths, adj_data, half_n_rel * 2 + 1
    return concept_ids, node_type_ids, adj_lengths, emb_data, adj_data, half_n_rel * 2 + 1


def load_gpt_input_tensors(statement_jsonl_path, max_seq_length):
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def load_qa_dataset(dataset_path):
        """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
        with open(dataset_path, "r", encoding="utf-8") as fin:
            output = []
            for line in fin:
                input_json = json.loads(line)
                label = ord(input_json.get("answerKey", "A")) - ord("A")
                output.append((input_json['id'], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]], label))
        return output

    def pre_process_datasets(encoded_datasets, num_choices, max_seq_length, start_token, delimiter_token, clf_token):
        """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

            To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
            input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        """
        tensor_datasets = []
        for dataset in encoded_datasets:
            n_batch = len(dataset)
            input_ids = np.zeros((n_batch, num_choices, max_seq_length), dtype=np.int64)
            mc_token_ids = np.zeros((n_batch, num_choices), dtype=np.int64)
            lm_labels = np.full((n_batch, num_choices, max_seq_length), fill_value=-1, dtype=np.int64)
            mc_labels = np.zeros((n_batch,), dtype=np.int64)
            for i, data, in enumerate(dataset):
                q, mc_label = data[0], data[-1]
                choices = data[1:-1]
                for j in range(len(choices)):
                    _truncate_seq_pair(q, choices[j], max_seq_length - 3)
                    qa = [start_token] + q + [delimiter_token] + choices[j] + [clf_token]
                    input_ids[i, j, :len(qa)] = qa
                    mc_token_ids[i, j] = len(qa) - 1
                    lm_labels[i, j, :len(qa) - 1] = qa[1:]
                mc_labels[i] = mc_label
            all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
            tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
        return tensor_datasets

    def tokenize_and_encode(tokenizer, obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        else:
            return list(tokenize_and_encode(tokenizer, o) for o in obj)

    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(GPT_SPECIAL_TOKENS)

    dataset = load_qa_dataset(statement_jsonl_path)
    examples_ids = [data[0] for data in dataset]
    dataset = [data[1:] for data in dataset]  # discard example ids
    num_choices = len(dataset[0]) - 2

    encoded_dataset = tokenize_and_encode(tokenizer, dataset)

    (input_ids, mc_token_ids, lm_labels, mc_labels), = pre_process_datasets([encoded_dataset], num_choices, max_seq_length, *special_tokens_ids)
    return examples_ids, mc_labels, input_ids, mc_token_ids, lm_labels


def get_gpt_token_num():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    return len(tokenizer)


def is_pad_answer(answer):
    return '--PAD ANSWER--'.lower() in answer.lower()


def load_bert_xlnet_roberta_input_tensors(statement_jsonl_path, model_type, model_name, max_seq_length,
                                          format=None, ret_num_labs=False):
    if format is None:
        format = []

    class InputExample(object):
        def __init__(self, example_id, question, contexts, endings, label=None):
            self.example_id = example_id
            self.question = question
            self.contexts = contexts
            self.endings = endings
            self.label = label

    class InputFeatures(object):
        def __init__(self, example_id, choices_features, label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for _, input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.label = label

    def read_examples(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                label = ord(json_dic["answerKey"]) - ord("A") if 'answerKey' in json_dic else 0
                contexts = json_dic["question"]["stem"]
                if "para" in json_dic:
                    contexts = json_dic["para"] + " " + contexts
                if "fact1" in json_dic:
                    contexts = json_dic["fact1"] + " " + contexts
                examples.append(
                    InputExample(
                        example_id=json_dic["id"],
                        contexts=[contexts] * len(json_dic["question"]["choices"]),
                        question="",
                        endings=[ending["text"] for ending in json_dic["question"]["choices"]],
                        label=label
                    ))
        return examples

    def convert_examples_to_features(examples,
                                     label_list,
                                     max_seq_length,
                                     tokenizer,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     sep_token_extra=False,
                                     pad_token_segment_id=0,
                                     pad_on_left=False,
                                     pad_token=0,
                                     mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}

        cls_token_segment_id = tokenizer.cls_token_id
        pad_token_segment_id = tokenizer.pad_token_id

        features = []
        for ex_index, example in enumerate(examples):
            choices_features = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                extra_args = {'add_prefix_space': True} if (model_type in ['roberta', 'LIAMF-USP___aristo-roberta'] and 'add_prefix_space' in format) else {}
                tokens_a = tokenizer.tokenize(context, **extra_args)
                tokens_b = tokenizer.tokenize(example.question + " " + ending, **extra_args)

                ending_is_pad = is_pad_answer(ending)

                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

                # The convention in BERT is:
                # (a) For sequence pairs:
                #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
                #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
                # (b) For single sequences:
                #  tokens:   [CLS] the dog is hairy . [SEP]
                #  type_ids:   0   0   0   0  0     0   0
                #
                # Where "type_ids" are used to indicate whether this is the first
                # sequence or the second sequence. The embedding vectors for `type=0` and
                # `type=1` were learned during pre-training and are added to the wordpiece
                # embedding vector (and position vector). This is not *strictly* necessary
                # since the [SEP] token unambiguously separates the sequences, but it makes
                # it easier for the model to learn the concept of sequences.
                #
                # For classification tasks, the first vector (corresponding to [CLS]) is
                # used as as the "sentence vector". Note that this only makes sense because
                # the entire model is fine-tuned.
                tokens = tokens_a + [sep_token]
                if sep_token_extra:
                    # roberta uses an extra separator b/w pairs of sentences
                    tokens += [sep_token]

                segment_ids = [sequence_a_segment_id] * len(tokens)

                if tokens_b:
                    tokens += tokens_b + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

                if cls_token_at_end:
                    tokens = tokens + [cls_token]
                    segment_ids = segment_ids + [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.

                input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                special_token_id = tokenizer.convert_tokens_to_ids([cls_token, sep_token])
                # Used only for kvmem
                output_mask = [1 if id in special_token_id else 0 for id in input_ids]  # 1 for mask

                # Zero-pad up to the sequence length.
                padding_length = max_seq_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                    output_mask = ([1] * padding_length) + output_mask

                    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    output_mask = output_mask + ([1] * padding_length)
                    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

                if ending_is_pad:
                    output_mask = [0] * max_seq_length
                    input_mask = [0 if mask_padding_with_zero else 1] * max_seq_length
                    segment_ids = [pad_token_segment_id] * max_seq_length

                assert len(input_ids) == max_seq_length
                assert len(output_mask) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_features.append((tokens, input_ids, input_mask, segment_ids, output_mask))
            label = label_map[example.label]
            features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label))

        return features

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_output_mask, all_label

    tokenizer_classes = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer,
                         'roberta': RobertaTokenizer, 'albert': AlbertTokenizer}

    if model_type in tokenizer_classes:
        tokenizer_class = tokenizer_classes.get(model_type)
        tokenizer = tokenizer_class.from_pretrained(model_name.replace("___", "/"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name.replace("___", "/"))
    examples = read_examples(statement_jsonl_path)

    if any(x in format for x in ('add_qa_prefix', 'fairseq')):
        for example in examples:
            example.contexts = ['Q: ' + c for c in example.contexts]
            example.endings = ['A: ' + e for e in example.endings]

    n_labels = 0
    for example in examples:
        n_labels = max(n_labels, len(example.endings))

    features = convert_examples_to_features(examples=examples,
                                            label_list=list(range(n_labels)),
                                            max_seq_length=max_seq_length,
                                            tokenizer=tokenizer,
                                            cls_token_at_end=bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta', 'LIAMF-USP___aristo-roberta',
                                                                                'albert']
                                                                 and 'no_extra_sep' not in format
                                                                 and 'fairseq' not in format),
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token=tokenizer.pad_token_id or 0,
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta',
                                                                                      'LIAMF-USP___aristo-roberta',
                                                                                      'albert'] or 'minilm' in model_type.lower() else 1)
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = convert_features_to_tensors(features)

    if ret_num_labs:
        num_labs = {}
        for example in examples:
            ex_id = example.example_id
            n_endings = 0
            for ending in example.endings:
                if not is_pad_answer(ending):
                    n_endings += 1
            num_labs[ex_id] = n_endings
        return example_ids, all_label, num_labs, *data_tensors
    else:
        return example_ids, all_label, *data_tensors


# def load_lstm_input_tensors(input_jsonl_path, max_seq_length):
#     def _truncate_seq_pair(tokens_a, tokens_b, max_length):
#         while len(tokens_a) + len(tokens_b) > max_length:
#             tokens_a.pop() if len(tokens_a) > len(tokens_b) else tokens_b.pop()
#
#     tokenizer = WordTokenizer.from_pretrained('lstm')
#     qids, labels, input_ids, input_lengths = [], [], [], []
#     pad_id, = tokenizer.convert_tokens_to_ids([PAD_TOK])
#     with open(input_jsonl_path, "r", encoding="utf-8") as fin:
#         for line in fin:
#             input_json = json.loads(line)
#             qids.append(input_json['id'])
#             labels.append(ord(input_json.get("answerKey", "A")) - ord("A"))
#             instance_input_ids, instance_input_lengths = [], []
#             question_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_json["question"]["stem"]))
#             for ending in input_json["question"]["choices"]:
#                 question_ids_copy = question_ids.copy()
#                 answer_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ending["text"]))
#                 _truncate_seq_pair(question_ids_copy, answer_ids, max_seq_length)
#                 ids = question_ids_copy + answer_ids + [pad_id] * (max_seq_length - len(question_ids_copy) - len(answer_ids))
#                 instance_input_ids.append(ids)
#                 instance_input_lengths.append(len(question_ids_copy) + len(answer_ids))
#             input_ids.append(instance_input_ids)
#             input_lengths.append(instance_input_lengths)
#     labels = torch.tensor(labels, dtype=torch.long)
#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     input_lengths = torch.tensor(input_lengths, dtype=torch.long)
#     return qids, labels, input_ids, input_lengths


def load_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length, format=None, ret_num_labs=False):
    if format is None:
        format = []
    if model_type in ('gpt',):
        return load_gpt_input_tensors(input_jsonl_path, max_seq_length)
    elif model_type in ('bert', 'xlnet', 'roberta', 'albert', 'distilbert') or '___' in model_type:
        return load_bert_xlnet_roberta_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length,
                                                     format=format, ret_num_labs=ret_num_labs)
    else:
        raise ValueError(f"Unknown model type {model_type}")


def load_statement_dict(statement_path):
    all_dict = {}
    with open(statement_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            instance_dict = json.loads(line)
            qid = instance_dict['id']
            all_dict[qid] = {
                'question': instance_dict['question']['stem'],
                'answers': [dic['text'] for dic in instance_dict['question']['choices']]
            }
    return all_dict
