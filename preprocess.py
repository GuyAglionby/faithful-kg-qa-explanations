import argparse
import os
from collections import defaultdict
from multiprocessing import cpu_count

from utils.conceptnet import construct_graph, extract_english
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.graph import generate_adj_data_from_grounded_concepts, generate_adj_data_from_grounded_concepts__use_lm
from utils.grounding import create_matcher_patterns, ground
from utils.tokenization_utils import tokenize_statement_file, make_word_vocab


def main():
    default_seed = 0xdefa014
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='graph', choices=['graph', 'csqa', 'obqa'],
                        help="Which preprocessing to do. "
                             "'graph' to preprocess the graph; name of the dataset to preprocess that dataset.")
    parser.add_argument('--graph', type=str, default='cpnet', choices=['cpnet'], help='Name of knowledge graph to use')

    parser.add_argument('--path_prune_threshold', type=float, default=0.0, help='Threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='Maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='Number of processes to use')
    parser.add_argument('--seed', type=int, default=default_seed, help='Random seed')

    parser.add_argument('-k', type=int, default=2, help="k hops to use when finding adj graph (WT specific currently)")
    parser.add_argument('--max_eventual_edge_num', type=int, default=-1,
                        help="how many edges in the graph (after all edges added, not just from paths)")
    parser.add_argument('--use-lm-scoring', action='store_true', help="Score the relevance of each node using LM "
                                                                      "(applicable to QA-GNN)")
    args, _ = parser.parse_known_args()

    parser.add_argument('--save-string', type=str, default=args.graph, help='String to add to saved files')

    args = parser.parse_args()

    graph = args.graph
    dataset = args.run
    seed = args.seed if args.seed != default_seed else None

    if args.use_lm_scoring:
        generate_adj_data_fn = generate_adj_data_from_grounded_concepts__use_lm
    else:
        generate_adj_data_fn = generate_adj_data_from_grounded_concepts

    input_paths = {
        'graph': defaultdict(str),
        'csqa': {
            'train': './data/csqa/train_rand_split.jsonl',
            'dev': './data/csqa/dev_rand_split.jsonl',
            'test': './data/csqa/test_rand_split_no_answers.jsonl',
        },
        'obqa': {
            'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
            'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
            'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
        },
        'cpnet': {
            'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
        },
        f'transe-cpnet': {
            'ent': './data/transe-cpnet/glove.transe.sgd.ent.npy',
            'rel': './data/transe-cpnet/glove.transe.sgd.rel.npy',
        }
    }

    output_paths = {
        'CUSTOM_GRAPH': {
            'csv': f'./data/{graph}/{graph}-mhgrn-format.tsv',
            'vocab': f'./data/{graph}/entity_vocab.txt',
            'patterns': f'./data/{graph}/matcher_patterns.json',
            'unpruned-graph': f'./data/{graph}/{graph}.en.unpruned.graph',
            'pruned-graph': f'./data/{graph}/{graph}.en.pruned.graph',
        },
        'graph': defaultdict(lambda: defaultdict(defaultdict)),
        # 'glove': {
        #     'npy': './data/glove/glove.6B.300d.npy',
        #     'vocab': './data/glove/glove.vocab',
        # },
        # 'numberbatch': {
        #     'npy': f'./data/transe/nb.npy',
        #     'vocab': f'./data/transe/nb.vocab',
        #     'concept_npy': f'./data/transe/concept.nb.npy'
        # },
        'dataset': {
            'statement': {
                'train': f'./data/{dataset}-{args.save_string}/statement/train.statement.jsonl',
                'dev': f'./data/{dataset}-{args.save_string}/statement/dev.statement.jsonl',
                'test': f'./data/{dataset}-{args.save_string}/statement/test.statement.jsonl',
                'train-fairseq': f'./data/{dataset}-{args.save_string}/fairseq/official/train.jsonl',
                'dev-fairseq': f'./data/{dataset}-{args.save_string}/fairseq/official/valid.jsonl',
                'test-fairseq': f'./data/{dataset}-{args.save_string}/fairseq/official/test.jsonl',
                'vocab': f'./data/{dataset}-{args.save_string}/statement/vocab.json',
            },
            'tokenized': {
                'train': f'./data/{dataset}-{args.save_string}/tokenized/train.tokenized.txt',
                'dev': f'./data/{dataset}-{args.save_string}/tokenized/dev.tokenized.txt',
                'test': f'./data/{dataset}-{args.save_string}/tokenized/test.tokenized.txt',
            },
            'grounded': {
                'train': f'./data/{dataset}-{args.save_string}/grounded/train.grounded.jsonl',
                'dev': f'./data/{dataset}-{args.save_string}/grounded/dev.grounded.jsonl',
                'test': f'./data/{dataset}-{args.save_string}/grounded/test.grounded.jsonl',
                # 'train-ids': f'./data/{dataset}-{args.save_string}/grounded/train.grounded-ids.txt',
                # 'dev-ids': f'./data/{dataset}-{args.save_string}/grounded/dev.grounded-ids.txt',
                # 'test-ids': f'./data/{dataset}-{args.save_string}/grounded/test.grounded-ids.txt',
            },
            'graph': {
                # 'train': f'./data/{dataset}-{args.save_string}/graph/train.graph.jsonl',
                # 'dev': f'./data/{dataset}-{args.save_string}/graph/dev.graph.jsonl',
                # 'test': f'./data/{dataset}-{args.save_string}/graph/test.graph.jsonl',
                'adj-train': f'./data/{dataset}-{args.save_string}/graph/train.graph.adj.pk',
                'adj-dev': f'./data/{dataset}-{args.save_string}/graph/dev.graph.adj.pk',
                'adj-test': f'./data/{dataset}-{args.save_string}/graph/test.graph.adj.pk',
                # 'nxg-from-adj-train': f'./data/{dataset}-{args.save_string}/graph/train.graph.adj.jsonl',
                # 'nxg-from-adj-dev': f'./data/{dataset}-{args.save_string}/graph/dev.graph.adj.jsonl',
                # 'nxg-from-adj-test': f'./data/{dataset}-{args.save_string}/graph/test.graph.adj.jsonl',
            },
        },
    }

    dataset_specific_conversion_function = {
        "csqa": convert_to_entailment,
        "obqa": convert_to_obqa_statement,
        "graph": ""
    }[dataset]

    verb_nominalisation_cache_file = './data/verb_nominalisation_cache_file.json'
    routines = {
        'graph': [
            # Keep this as just for conceptnet only - we don't need to extract english only for our other graphs.
            # Just need to ensure that pre-made files are at the specifid output locations
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['CUSTOM_GRAPH']['csv'],
                                               output_paths['CUSTOM_GRAPH']['vocab'])},
            {'func': construct_graph,
             'args': (output_paths['CUSTOM_GRAPH']['csv'], output_paths['CUSTOM_GRAPH']['vocab'],
                      output_paths['CUSTOM_GRAPH']['unpruned-graph'], graph, False)},
            {'func': construct_graph,
             'args': (output_paths['CUSTOM_GRAPH']['csv'], output_paths['CUSTOM_GRAPH']['vocab'],
                      output_paths['CUSTOM_GRAPH']['pruned-graph'], graph, True)},
            {'func': create_matcher_patterns, 'args': (output_paths['CUSTOM_GRAPH']['vocab'],
                                                       output_paths['CUSTOM_GRAPH']['patterns'],
                                                       graph, verb_nominalisation_cache_file)},
        ],
        'dataset': [
            # Converting from original dataset format to our format
            {'func': dataset_specific_conversion_function, 'args': (input_paths[dataset]['train'],
                                                                    output_paths['dataset']['statement']['train'],
                                                                    output_paths['dataset']['statement'][
                                                                        'train-fairseq'])},
            {'func': dataset_specific_conversion_function, 'args': (input_paths[dataset]['dev'],
                                                                    output_paths['dataset']['statement']['dev'],
                                                                    output_paths['dataset']['statement'][
                                                                        'dev-fairseq'])},
            {'func': dataset_specific_conversion_function, 'args': (input_paths[dataset]['test'],
                                                                    output_paths['dataset']['statement']['test'],
                                                                    output_paths['dataset']['statement'][
                                                                        'test-fairseq'])},
            # Tokenizing
            {'func': tokenize_statement_file, 'args': (output_paths['dataset']['statement']['train'],
                                                       output_paths['dataset']['tokenized']['train'])},
            {'func': tokenize_statement_file, 'args': (output_paths['dataset']['statement']['dev'],
                                                       output_paths['dataset']['tokenized']['dev'])},
            {'func': tokenize_statement_file, 'args': (output_paths['dataset']['statement']['test'],
                                                       output_paths['dataset']['tokenized']['test'])},
            {'func': make_word_vocab, 'args': ((output_paths['dataset']['statement']['train'],),
                                               output_paths['dataset']['statement']['vocab'])},
            # Grounding (entity linking)
            {'func': ground, 'args': (output_paths['dataset']['statement']['train'],
                                      output_paths['CUSTOM_GRAPH']['vocab'],
                                      output_paths['CUSTOM_GRAPH']['patterns'],
                                      output_paths['dataset']['grounded']['train'],
                                      output_paths['CUSTOM_GRAPH']['pruned-graph'],
                                      args.nprocs, verb_nominalisation_cache_file)},
            {'func': ground, 'args': (output_paths['dataset']['statement']['dev'],
                                      output_paths['CUSTOM_GRAPH']['vocab'],
                                      output_paths['CUSTOM_GRAPH']['patterns'],
                                      output_paths['dataset']['grounded']['dev'],
                                      output_paths['CUSTOM_GRAPH']['pruned-graph'],
                                      args.nprocs, verb_nominalisation_cache_file)},
            {'func': ground, 'args': (output_paths['dataset']['statement']['test'],
                                      output_paths['CUSTOM_GRAPH']['vocab'],
                                      output_paths['CUSTOM_GRAPH']['patterns'],
                                      output_paths['dataset']['grounded']['test'],
                                      output_paths['CUSTOM_GRAPH']['pruned-graph'],
                                      args.nprocs, verb_nominalisation_cache_file)},
            # Generating input graphs for each question (subgraph of the full KG; sometimes known as 'schema graph')
            {'func': generate_adj_data_fn, 'args': (output_paths['dataset']['grounded']['train'],
                                                    output_paths['CUSTOM_GRAPH']['pruned-graph'],
                                                    output_paths['CUSTOM_GRAPH']['vocab'],
                                                    output_paths['dataset']['graph']['adj-train'],
                                                    args.nprocs, graph, args.k,
                                                    args.max_node_num, args.max_eventual_edge_num, seed)},
            {'func': generate_adj_data_fn, 'args': (output_paths['dataset']['grounded']['dev'],
                                                    output_paths['CUSTOM_GRAPH']['pruned-graph'],
                                                    output_paths['CUSTOM_GRAPH']['vocab'],
                                                    output_paths['dataset']['graph']['adj-dev'],
                                                    args.nprocs, graph, args.k,
                                                    args.max_node_num, args.max_eventual_edge_num, seed)},
            {'func': generate_adj_data_fn, 'args': (output_paths['dataset']['grounded']['test'],
                                                    output_paths['CUSTOM_GRAPH']['pruned-graph'],
                                                    output_paths['CUSTOM_GRAPH']['vocab'],
                                                    output_paths['dataset']['graph']['adj-test'],
                                                    args.nprocs, graph, args.k,
                                                    args.max_node_num, args.max_eventual_edge_num, seed)},
        ]
    }

    if args.run != 'graph':
        base_dir = f"data/{args.run}-{graph}"
        # os.makedirs(f"{base_dir}/fairseq/official", exist_ok=True)
        # os.makedirs(f"{base_dir}/fairseq/inhouse", exist_ok=True)
        os.makedirs(f"{base_dir}/grounded/", exist_ok=True)
        os.makedirs(f"{base_dir}/graph/", exist_ok=True)
        os.makedirs(f"{base_dir}/statement/", exist_ok=True)
        os.makedirs(f"{base_dir}/tokenized/", exist_ok=True)
        # os.makedirs(f"{base_dir}/roberta/", exist_ok=True)

    suite = args.run if args.run == 'graph' else 'dataset'

    for rt_dic in routines[suite]:
        rt_dic['func'](*rt_dic['args'])

    print(f'Successfully run {args.run}')


if __name__ == '__main__':
    main()
