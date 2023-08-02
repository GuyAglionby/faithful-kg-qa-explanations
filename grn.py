import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from modeling.modeling_grn import LMGraphRelationNetDataLoader, LMGraphRelationNet
from utils.conceptnet import merged_relations
from utils.data_utils import load_statement_dict
from utils.optimization_utils import OPTIMIZER_CLASSES, masked_log_softmax
from utils.parser_utils import get_parser, get_lstm_config_from_args
from utils.utils import bool_flag, export_config, check_path, unfreeze_net, freeze_net

DECODER_DEFAULT_LR = {
    'csqa': {
        'cpnet': 1e-3
    },
    'obqa': {
        'cpnet': 1e-3,
    },
    'worldtree': {
        'cpnet': 3e-4,
    },
}

logger = logging.getLogger(__name__)


def get_node_feature_encoder(encoder_name):
    return encoder_name.replace('-cased', '-uncased')


def evaluate_accuracy(eval_set, n_labs, model, return_preds=False):
    n_samples, n_correct = 0, 0
    model.eval()
    predictions = []
    with torch.no_grad():
        for qids, labels, *input_data in eval_set:
            logits, _ = model(*input_data)
            logits_min = logits.min() - 5
            if n_labs is not None:
                softmax_mask = torch.zeros(logits.shape).bool()
                for i, qid in enumerate(qids):
                    softmax_mask[i, n_labs[qid]:] = True
                logits[softmax_mask] = logits_min
            logits_argmax = logits.argmax(1)
            for qid, index in zip(qids, logits_argmax.tolist()):
                predictions.append('{},{}'.format(qid, chr(ord('A') + index)))
            n_correct += (logits_argmax == labels).sum().item()
            n_samples += labels.size(0)
    if return_preds:
        return n_correct / n_samples, predictions
    return n_correct / n_samples


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'decode'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/grn/', help='model output directory')
    parser.add_argument('--save_optimizer', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--load_model_path', default=None)
    parser.add_argument('--load_optimizer_path', default=None)

    # data
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--train_adj', default=f'./data/{args.dataset}-{args.graph}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'./data/{args.dataset}-{args.graph}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'./data/{args.dataset}-{args.graph}/graph/test.graph.adj.pk')
    # parser.add_argument('--train_embs', default=f'./data/{args.dataset}-{args.graph}/features/train.{get_node_feature_encoder(args.encoder)}.features.pk')
    # parser.add_argument('--dev_embs', default=f'./data/{args.dataset}-{args.graph}/features/dev.{get_node_feature_encoder(args.encoder)}.features.pk')
    # parser.add_argument('--test_embs', default=f'./data/{args.dataset}-{args.graph}/features/test.{get_node_feature_encoder(args.encoder)}.features.pk')

    # model architecture
    parser.add_argument('-k', '--k', default=2, type=int, help='perform k-hop message passing at each layer')
    parser.add_argument('--ablation', default=[], choices=['no_trans', 'early_relu', 'no_att', 'ctx_trans', 'q2a_only',
                                                           'no_typed_transform', 'no_type_att', 'typed_pool', 'no_unary',
                                                           'detach_s_agg', 'detach_s_all', 'detach_s_pool', 'agg_self_loop',
                                                           'early_trans', 'pool_qc', 'pool_ac', 'pool_all',
                                                           'no_ent', 'no_rel', 'no_rel_att', 'no_1hop', 'fix_scale',
                                                           'no_lm', 'no_s_in_final_mlp', 'no_grn',
                                                           'no_attentional_aggregation'],
                        nargs='*', help='run ablation test')
    parser.add_argument('-dd', '--diag_decompose', default=True, type=bool_flag, nargs='?', const=True, help='use diagonal decomposition')
    parser.add_argument('--num_basis', default=0, type=int, help='number of basis (0 to disable basis decomposition)')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--att_dim', default=50, type=int, help='dimensionality of the query vectors')
    parser.add_argument('--att_layer_num', default=1, type=int, help='number of hidden layers of the attention module')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--gnn_layer_num', default=1, type=int, help='number of GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
    parser.add_argument('--eps', type=float, default=1e-15, help='avoid numeric overflow')
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--init_rn', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--init_identity', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--fix_trans', default=False, type=bool_flag, nargs='?', const=True)

    # regularization
    parser.add_argument('--dropouti', type=float, default=0.1, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.1, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')
    parser.add_argument('--groupnorm', action='store_true')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset].get(args.graph, DECODER_DEFAULT_LR[args.dataset]['cpnet']), type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int,
                        help="basically as an alternative to grad accumulation (NOT minibatch as usually defined)")
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(diag_decompose=True, gnn_layer_num=1, k=1)

    graph_relation_list = f"data/{args.graph}/relations.tsv"
    if os.path.exists(graph_relation_list):
        with open(graph_relation_list) as f:
            lines = [l.strip() for l in f if len(l.strip())]
            num_relation = len(lines) * 2
            parser.set_defaults(num_relation=num_relation)
    elif args.graph == 'cpnet':
        parser.set_defaults(num_relation=len(merged_relations) * 2)

    args = parser.parse_args()

    if os.path.exists(args.save_dir) and args.mode == 'train':
        raise ValueError(f"Save dir '{args.save_dir}' already exists!")

    logfile = args.save_dir + "/output.log"
    check_path(logfile)
    logging.basicConfig(filename=logfile,
                        filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    writer = SummaryWriter(f"{args.save_dir}/tensorboard")

    if args.mode == 'train':
        logger.info("**** Training ****")
        train(args, writer)
    elif args.mode == 'eval':
        logger.info("**** Evaluating ****")
        eval(args)
    elif args.mode == 'decode':
        logger.info("**** Decoding ****")
        decode(args)
    else:
        raise ValueError('Invalid mode')

    writer.close()


def train(args, writer):
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    optimizer_path = os.path.join(args.save_dir, 'optimizer.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,dev_acc,test_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    # if 'lm' in args.ent_emb:
    #     logger.info('Using contextualized embeddings for concepts')
    #     use_contextualized = True
    # else:
    use_contextualized = False

    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    logger.info('| num_concepts: %d |', concept_num)

    try:
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
        dataset = LMGraphRelationNetDataLoader(args.train_statements, args.train_adj,
                                               args.dev_statements, args.dev_adj,
                                               args.test_statements, args.test_adj,
                                               batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=(device0, device1),
                                               model_name=args.encoder,
                                               max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                               is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                               use_contextualized=False,
                                               train_embs_path=None, dev_embs_path=None, test_embs_path=None,
                                               subsample=args.subsample, format=args.format)

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################

        if args.load_model_path:
            logger.info("Loading model...")
            model, _ = torch.load(args.load_model_path)
        else:
            logger.info("Initialising new model...")
            lstm_config = get_lstm_config_from_args(args)
            model = LMGraphRelationNet(args.encoder, k=args.k, n_type=3, n_basis=args.num_basis, n_layer=args.gnn_layer_num,
                                       diag_decompose=args.diag_decompose, n_concept=concept_num,
                                       n_relation=args.num_relation, concept_dim=args.gnn_dim,
                                       concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
                                       n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                                       att_dim=args.att_dim, att_layer_num=args.att_layer_num,
                                       p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                                       pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                                       ablation=args.ablation, init_range=args.init_range,
                                       eps=args.eps, use_contextualized=use_contextualized,
                                       do_init_rn=args.init_rn, do_init_identity=args.init_identity,
                                       encoder_config=lstm_config, groupnorm=args.groupnorm, seed=args.seed)
        model.encoder.to(device0)
        model.decoder.to(device1)
    except RuntimeError as e:
        logger.info(e)
        logger.info('best dev acc: 0.0 (at epoch 0)')
        logger.info('final test acc: 0.0')
        logger.info("")
        return

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.fix_trans:
        no_decay.append('trans_scores')
    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    if args.load_optimizer_path:
        logger.info("Loading optimizer...")
        optimizer = torch.load(args.load_optimizer_path)
    else:
        logger.info("Initialising new optimizer...")
        optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=max_steps)
    else:
        raise ValueError("Invalid LR schedule:", args.lr_schedule)

    logger.info('parameters (decoder only):')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            logger.info('\t%.45s\ttrainable\t%s', name, param.size())
        else:
            logger.info('\t%.45s\tfixed\t%s', name, param.size())
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    logger.info('\ttotal: %d', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        # loss_func = nn.CrossEntropyLoss(reduction='mean')
        loss_func = nn.NLLLoss(reduction='mean')
    else:
        raise ValueError(f"Invalid loss function:", args.loss)

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    logger.info("")
    logger.info("-" * 71)
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    freeze_net(model.encoder)
    try:
        for epoch_id in range(args.n_epochs):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
            model.train()
            for qids, labels, *input_data in dataset.train():
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)

                    if args.loss == 'margin_rank':
                        logger.warning("Masking for different # of MCQ cands not implemented")
                        num_choice = logits.size(1)
                        flat_logits = logits.view(-1)
                        correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
                        correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
                        wrong_logits = flat_logits[correct_mask == 0]  # of length batch_size*(num_choice-1)
                        y = wrong_logits.new_ones((wrong_logits.size(0),))
                        loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
                    elif args.loss == 'cross_entropy':
                        softmax_mask = torch.ones((logits.shape[0], logits.shape[1])).bool()
                        for i, qid in enumerate(qids[a:b]):
                            n_labs = dataset.train_n_labs[qid]
                            softmax_mask[i, n_labs:] = False
                        log_softmax = masked_log_softmax(logits, mask=softmax_mask.to(logits.device))
                        loss = loss_func(log_softmax, labels[a:b])
                    else:
                        raise ValueError("Invalid loss function")
                    loss = loss * (b - a) / bs
                    loss.backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                global_step += 1
                if global_step % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    logger.info("| e %.3d | step %.5d | lr %9.7f | loss %7.4f | ms/batch %7.2f |", epoch_id, global_step, scheduler.get_lr()[0], total_loss, ms_per_batch)
                    writer.add_scalar("training loss", total_loss, global_step)
                    writer.add_scalar("training ms per batch", ms_per_batch, global_step)
                    for torch_device_i in range(torch.cuda.device_count()):
                        writer.add_scalar(f"GPU{torch_device_i} memory reserved",
                                          torch.cuda.memory_reserved(torch_device_i))
                        writer.add_scalar(f"GPU{torch_device_i} memory allocated",
                                          torch.cuda.memory_allocated(torch_device_i))
                    total_loss = 0
                    start_time = time.time()

            model.eval()
            dev_acc, dev_preds = evaluate_accuracy(dataset.dev(),
                                                   dataset.dev_n_labs if args.dataset != 'csqa' else None,
                                                   model, return_preds=True)
            test_acc, test_preds = evaluate_accuracy(dataset.test(),
                                                     dataset.test_n_labs if args.dataset != 'csqa' else None,
                                                     model, return_preds=True) if dataset.test_size() else (0.0, [])
            with open(f"{args.save_dir}/test_e{epoch_id}_preds.csv", "w") as f:
                f.write("\n".join(test_preds))
            with open(f"{args.save_dir}/dev_e{epoch_id}_preds.csv", "w") as f:
                f.write("\n".join(dev_preds))

            logger.info('-' * 71)
            writer.add_scalar("dev acc", dev_acc, global_step)
            writer.add_scalar("test acc", test_acc, global_step)
            logger.info('| e %.3d | step %.5d | dev_acc %7.4f | test_acc %7.4f |', epoch_id, global_step, dev_acc, test_acc)
            logger.info('-' * 71)
            with open(log_path, 'a') as fout:
                fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                if args.save_model:
                    logger.info('saving model...')
                    torch.save([model, args], model_path)
                    logger.info('model saved to %s', model_path)
                if args.save_optimizer:
                    logger.info('saving optimizer...')
                    torch.save(optimizer, optimizer_path)
                    logger.info('optimizer saved to %s', optimizer_path)
            model.train()
            start_time = time.time()
            if epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break
    except KeyboardInterrupt as _:
        logger.info("Keyboard interrupt")

    logger.info("")
    logger.info('training ends in %d steps', global_step)
    logger.info('best dev acc: %.4f (at epoch %d)', best_dev_acc, best_dev_epoch)
    logger.info('final test acc: %.4f', final_test_acc)
    logger.info("")


def eval(args):
    model_path = os.path.join(args.save_dir, 'model.pt')
    model, old_args = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.eval()

    # use_contextualized = 'lm' in old_args.ent_emb
    dataset = LMGraphRelationNetDataLoader(args.train_statements, args.train_adj,
                                           args.dev_statements, args.dev_adj,
                                           args.test_statements, args.test_adj,
                                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                           device=(device, device),
                                           model_name=old_args.encoder,
                                           max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                                           is_inhouse=old_args.inhouse,
                                           inhouse_train_qids_path=old_args.inhouse_train_qids,
                                           use_contextualized=False,
                                           train_embs_path=None, dev_embs_path=None, test_embs_path=None,
                                           subsample=old_args.subsample, format=old_args.format)

    logger.info("")
    logger.info("***** running evaluation *****")
    logger.info("| dataset: %s | num_dev: %d | num_test: %d | save_dir: %s",
                old_args.dataset, dataset.dev_size(), dataset.test_size(), args.save_dir)

    dev_acc, dev_preds = evaluate_accuracy(dataset.dev(),
                                           dataset.dev_n_labs if args.dataset != 'csqa' else None,
                                           model, return_preds=True)
    test_acc, test_preds = evaluate_accuracy(dataset.test(),
                                             dataset.test_n_labs if args.dataset != 'csqa' else None,
                                             model, return_preds=True) if dataset.test_size() else 0.0
    with open(f"{args.save_dir}/final_test_preds.csv", "w") as f:
        f.write("\n".join(test_preds))
    with open(f"{args.save_dir}/final_dev_preds.csv", "w") as f:
        f.write("\n".join(dev_preds))
    logger.info("***** evaluation done *****")
    logger.info("")
    logger.info('| dev_accuracy: %f | test_acc: %f |', dev_acc, test_acc)


def decode(args):
    model_path = os.path.join(args.save_dir, 'model.pt')
    model, old_args = torch.load(model_path, map_location='cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.eval()

    statement_dic = {}
    for statement_path in (old_args.train_statements, old_args.dev_statements, old_args.test_statements):
        statement_dic.update(load_statement_dict(statement_path))

    # use_contextualized = 'lm' in old_args.ent_emb
    dataset = LMGraphRelationNetDataLoader(old_args.train_statements, old_args.train_adj,
                                           old_args.dev_statements, old_args.dev_adj,
                                           old_args.test_statements, old_args.test_adj,
                                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=(device, device),
                                           model_name=old_args.encoder,
                                           max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                                           is_inhouse=old_args.inhouse, inhouse_train_qids_path=old_args.inhouse_train_qids,
                                           use_contextualized=False,
                                           train_embs_path=None, dev_embs_path=None, test_embs_path=None,
                                           subsample=old_args.subsample, format=old_args.format)

    with open(args.cpnet_vocab_path, 'r', encoding='utf-8') as fin:
        id2concept = [w.strip() for w in fin]

    if args.graph == 'cpnet':
        id2relation = merged_relations
    else:
        relation_vocab_path = args.cpnet_vocab_path.replace("entity_vocab.txt", "relations.tsv")
        with open(relation_vocab_path) as f:
            id2relation = [x.split("\t")[1].strip() for x in f]

    def path_ids_to_text(path_ids):
        assert len(path_ids) % 2 == 1
        res = []
        for p in range(len(path_ids)):
            if p % 2 == 0:  # entity
                res.append(id2concept[path_ids[p].item()])
            else:  # relationi
                rid = path_ids[p].item()
                if rid < len(id2relation):
                    res.append('<--[{}]---'.format(id2relation[rid]))
                else:
                    res.append('---[{}]--->'.format(id2relation[rid - len(id2relation)]))
        return ' '.join(res)

    logger.info("| dataset: %s | num_dev: %d | num_test: %d | save_dir: %s |",
                old_args.dataset, dataset.dev_size(), dataset.test_size(), args.save_dir)
    model.eval()
    for eval_set, filename in zip([dataset.dev(), dataset.test()], ['decode_dev.txt', 'decode_test.txt']):
        outputs = []
        with torch.no_grad():
            for qids, labels, *input_data in tqdm(eval_set):
                logits, path_ids, path_lengths = model.decode(*input_data)
                logits_min = logits.min() - 5

                if 'dev' in filename:
                    n_labs = dataset.dev_n_labs
                elif 'test' in filename:
                    n_labs = dataset.test_n_labs
                else:
                    raise ValueError()

                softmax_mask = torch.zeros(logits.shape).bool()
                for i, qid in enumerate(qids):
                    softmax_mask[i, n_labs[qid]:] = True
                logits[softmax_mask] = logits_min

                predictions = logits.argmax(1)
                for i, (qid, label, pred) in enumerate(zip(qids, labels, predictions)):
                    outputs.append('*' * 60)
                    outputs.append('id: {}'.format(qid))
                    outputs.append('question: {}'.format(statement_dic[qid]['question']))
                    outputs.append('answer: {}'.format(statement_dic[qid]['answers'][label.item()]))
                    outputs.append('prediction: {}'.format(statement_dic[qid]['answers'][pred.item()]))
                    for j, answer in enumerate(statement_dic[qid]['answers']):
                        path = path_ids[i, j, :path_lengths[i, j]]
                        outputs.append('{:25} {}'.format('[{}. {}]{}{}'.format(chr(ord('A') + j),
                                                                               answer,
                                                                               '*' if j == label else '',
                                                                               '^' if j == pred else ''),
                                                         path_ids_to_text(path)))
        output_path = os.path.join(args.save_dir, filename)
        with open(output_path, 'w') as fout:
            for line in outputs:
                fout.write(line + '\n')
        logger.info('outputs saved to %s', output_path)
    logger.info("***** done *****")


if __name__ == '__main__':
    main()
