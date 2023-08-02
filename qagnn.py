import argparse
import datetime
import json
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from modeling.modeling_qagnn import LM_QAGNN_DataLoader, LM_QAGNN
from utils.conceptnet import merged_relations
from utils.optimization_utils import OPTIMIZER_CLASSES, masked_log_softmax
from utils.parser_utils import get_parser
from utils.utils import bool_flag, export_config, check_path, unfreeze_net, freeze_net


DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
    'worldtree': 3e-4
}

logger = logging.getLogger(__name__)


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


def build_parser():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/qagnn/', help='model output directory')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_optimizer', action='store_true')
    parser.add_argument('--load_model_path', default=None)
    parser.add_argument('--load_optimizer_path', default=None)

    # data
    parser.add_argument('--train_adj', default=f'data/{args.dataset}-{args.graph}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}-{args.graph}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}-{args.graph}/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True,
                        help='use cached data to accelerate data loading')

    # model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--ablation', default=[], choices=['no_s_in_final_mlp_direct',
                                                           'no_s_in_final_mlp_from_graph',
                                                           'no_node_scoring', 'detach_s_all',
                                                           'no_batchnorm'],
                        nargs='*', help='run ablation test')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads '
                                                                    '(for pooling, not message passing)')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True,
                        help='freeze entity embedding layer')

    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')

    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float,
                        help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)

    graph_relation_list = f"data/{args.graph}/relations.tsv"
    if os.path.exists(graph_relation_list):
        with open(f"data/{args.graph}/relations.tsv") as f:
            lines = [l.strip() for l in f if len(l.strip())]
            # + 2 for ctx->q and ctx->a
            # * 2 for reverse
            num_relation = (len(lines) + 2) * 2
            parser.set_defaults(num_relation=num_relation)
    elif args.graph == 'cpnet':
        parser.set_defaults(num_relation=(len(merged_relations) + 2) * 2)

    args = parser.parse_args()
    return args


def main():
    args = build_parser()

    if os.path.exists(args.save_dir) and args.mode == 'train':
        load_model_path_str = ' When loading an existing model, you must still provide a new save directory.'
        raise ValueError(
            f"Save dir '{args.save_dir}' already exists!{load_model_path_str if args.load_model_path else ''}")

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

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval_detail':
        eval_detail(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
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
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    logger.info('| num_concepts: %d |', concept_num)

    # try:
    if True:
        if torch.cuda.device_count() >= 2 and args.cuda:
            logger.info("2 GPU mode")
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
        elif torch.cuda.device_count() == 1 and args.cuda:
            logger.info("1 GPU mode")
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            logger.info("CPU mode")
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
        dataset = LM_QAGNN_DataLoader(args.train_statements, args.train_adj,
                                      args.dev_statements, args.dev_adj,
                                      args.test_statements, args.test_adj,
                                      batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                      device=(device0, device1),
                                      model_name=args.encoder,
                                      max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                      is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                      subsample=args.subsample, use_cache=args.use_cache)

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################

        if args.load_model_path:
            logger.info("Loading model...")
            model, _ = load_saved_model(args.load_model_path)
        else:
            logger.info("Initialising new QA-GNN model")
            model = LM_QAGNN(args.encoder, k=args.k, n_ntype=4, n_etype=args.num_relation, n_concept=concept_num,
                             concept_dim=args.gnn_dim,
                             concept_in_dim=concept_dim,
                             n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                             p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                             pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                             init_range=args.init_range,
                             encoder_config={}, ablation=args.ablation)

        model.encoder.to(device0)
        model.decoder.to(device1)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    if args.load_optimizer_path:
        logger.info("Loading optimizer...")
        optimizer = torch.load(args.load_optimizer_path)
    else:
        logger.info("Initialising new optimizer")
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

    logger.info('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            logger.info('\t%.45s\ttrainable\t%s\tdevice: %s', name, param.size(), param.device)
        else:
            logger.info('\t%.45s\tfixed\t%s\tdevice: %s', name, param.size(), param.device)
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
            if epoch_id == args.unfreeze_epoch and 'detach_s_all' not in args.ablation:
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
                        num_choice = logits.size(1)
                        flat_logits = logits.view(-1)
                        # of length batch_size*num_choice
                        correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)
                        # of length batch_size*(num_choice-1)
                        correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1,
                                                                                                        num_choice - 1).contiguous().view(
                            -1)
                        wrong_logits = flat_logits[correct_mask == 0]
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
                    logger.info("| e %.3d | step %.5d | lr %9.7f | loss %7.4f | ms/batch %7.2f |",
                                epoch_id, global_step, scheduler.get_lr()[0], total_loss, ms_per_batch)
                    total_loss = 0
                    start_time = time.time()

            model.eval()
            dev_acc, dev_preds = evaluate_accuracy(dataset.dev(),
                                                   dataset.dev_n_labs if args.dataset != 'csqa' else None,
                                                   model, return_preds=True)
            test_acc, test_preds = evaluate_accuracy(dataset.test(),
                                                     dataset.test_n_labs if args.dataset != 'csqa' else None,
                                                     model, return_preds=True) if dataset.test_size() else 0.0
            with open(f"{args.save_dir}/test_e{epoch_id}_preds.csv", "w") as f:
                f.write("\n".join(test_preds))
            with open(f"{args.save_dir}/dev_e{epoch_id}_preds.csv", "w") as f:
                f.write("\n".join(dev_preds))

            logger.info('-' * 71)
            logger.info('| e %.3d | step %.5d | dev_acc %7.4f | test_acc %7.4f |', epoch_id, global_step, dev_acc,
                        test_acc)
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
                    logger.info(f'model saved to {model_path}')

                if args.save_optimizer:
                    logger.info('saving optimizer...')
                    torch.save(optimizer, optimizer_path)
                    logger.info(f'optimizer saved to {optimizer_path}')
            model.train()
            start_time = time.time()
            if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break
    except KeyboardInterrupt as _:
        logger.info("Keyboard interrupt")

    logger.info("")
    logger.info('training ends in %d steps', global_step)
    logger.info('best dev acc: %.4f (at epoch %d)', best_dev_acc, best_dev_epoch)
    logger.info('final test acc: %.4f', final_test_acc)
    logger.info("")


def load_saved_model(path):
    # This is hacky - have to first make a new instance of LM_QAGNN, and then load the state dict in
    # This is because `torch.load` will load pickled versions of the various classes that comprise the model,
    # and the classes may have changed since train-time. Thus leading to breakages.
    assert os.path.exists(path), f"Provided model load path doesn't exist: {path}"
    loaded_model, old_args = torch.load(path, map_location=torch.device("cpu"))
    model_dir = "/".join(path.split("/")[:-1])
    with open(f"{model_dir}/config.json") as f:
        config = json.load(f)

    cp_emb = [np.load(path) for path in config['ent_emb_paths']]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

    model = LM_QAGNN(config['encoder'], k=config['k'], n_ntype=4, n_etype=config['num_relation'],
                     n_concept=concept_num,
                     concept_dim=config['gnn_dim'],
                     concept_in_dim=concept_dim,
                     n_attention_head=config['att_head_num'], fc_dim=config['fc_dim'],
                     n_fc_layer=config['fc_layer_num'],
                     p_emb=config['dropouti'], p_gnn=config['dropoutg'], p_fc=config['dropoutf'],
                     pretrained_concept_emb=cp_emb, freeze_ent_emb=config['freeze_ent_emb'],
                     init_range=config['init_range'],
                     encoder_config={}, ablation=config['ablation'])

    model.load_state_dict(loaded_model.state_dict())
    return model, old_args


def eval_detail(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path
    model, old_args = load_saved_model(model_path)

    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    model.encoder.to(device0)
    model.decoder.to(device1)
    model.eval()

    logger.info('inhouse? %s', args.inhouse)

    logger.info('args.train_statements: %s', args.train_statements)
    logger.info('args.dev_statements: %s', args.dev_statements)
    logger.info('args.test_statements: %s', args.test_statements)
    logger.info('args.train_adj %s', args.train_adj)
    logger.info('args.dev_adj %s', args.dev_adj)
    logger.info('args.test_adj %s', args.test_adj)

    dataset = LM_QAGNN_DataLoader(args.train_statements, old_args.train_adj,
                                  args.dev_statements, old_args.dev_adj,
                                  args.test_statements, old_args.test_adj,
                                  batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                  device=(device0, device1),
                                  model_name=old_args.encoder,
                                  max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                                  is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                  subsample=args.subsample, use_cache=args.use_cache)

    save_test_preds = args.save_model
    if not save_test_preds:
        test_acc = evaluate_accuracy(dataset.test(), dataset.test_n_labs, model) if args.test_statements else 0.0
        logger.info('-' * 71)
        logger.info('test_acc: %7.4f', test_acc)
        logger.info('-' * 71)
        dev_acc = evaluate_accuracy(dataset.dev(), dataset.dev_n_labs, model) if args.dev_statements else 0.0
        logger.info('-' * 71)
        logger.info('dev_acc: %7.4f', dev_acc)
        logger.info('-' * 71)
    else:
        logger.info('-' * 71)
        for set_name, eval_set, set_n_labs in [
            ('dev', dataset.dev(), dataset.dev_n_labs),
            ('test', dataset.test(), dataset.test_n_labs)
        ]:
            total_acc = []
            dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
            save_dir = "/".join(model_path.split("/")[:-1])
            preds_path = os.path.join(save_dir, f'{set_name}_preds_{dt}.csv')
            all_cached_outputs = []
            with open(preds_path, 'w') as f_preds:
                with torch.no_grad():
                    for qids, labels, *input_data in tqdm(eval_set):
                        logits, pooler_attn, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data,
                                                                                                       detail=True,
                                                                                                       cache_output=True)

                        # [0] because batch size 1
                        edge_type = [sum(x, []) for x in input_data[-1:]][0]
                        predictions = logits.argmax(1)  # [bsize, ]

                        logits_min = logits.min().cpu() - 5
                        logits_masked = logits.clone().cpu()
                        if set_n_labs is not None:
                            softmax_mask = torch.zeros(logits.shape).bool()
                            for i, qid in enumerate(qids):
                                softmax_mask[i, set_n_labs[qid]:] = True
                            logits_masked[softmax_mask] = logits_min

                        assert len(qids) == 1, "only works with eval batch size 1"
                        all_cached_outputs.append({
                            "qids": qids,
                            "concept_ids": model.decoder.concept_ids.cpu(),
                            "attention_weights": model.decoder.attention_weights,
                            "pooler_attn": pooler_attn,
                            "edgetype": [a.cpu() for a in edge_type],
                            "label": labels[0].item(),
                            "pred": predictions[0].item(),
                            "logits": logits.cpu(),
                            "logits_masked": logits_masked,
                        })

                        preds_ranked = (-logits).argsort(1)  # [bsize, n_choices]
                        for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(
                                zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index,
                                    edge_type)):
                            acc = int(pred.item() == label.item())
                            print('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                            f_preds.flush()
                            total_acc.append(acc)

                print(f"output to {os.path.join(save_dir, f'{set_name}_processed_attention_outputs.pkl')}")
                with open(os.path.join(save_dir, f"{set_name}_attention_outputs.pkl"), 'wb') as f:
                    pickle.dump(all_cached_outputs, f)
                acc = sum(total_acc) / len(total_acc)
                logger.info(f'{set_name}_acc: %7.4f', acc)
        logger.info('-' * 71)


if __name__ == '__main__':
    main()
