import itertools
import json
import logging
import os
import pickle
import random
import re
import string
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool

import networkx as nx
import numpy as np
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM

from .conceptnet import merged_relations

__all__ = [
    'generate_adj_data_from_grounded_concepts',
    'generate_adj_data_from_grounded_concepts__use_lm',
    'get_lm_score'
]

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None

logger = logging.getLogger(__name__)


def load_resources(cpnet_vocab_path, graph):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    if graph == 'cpnet':
        id2relation = merged_relations
    else:
        # relations.tsv from
        relation_vocab_path = cpnet_vocab_path.replace("entity_vocab.txt", "relations.tsv")
        with open(relation_vocab_path) as f:
            id2relation = [x.split("\t")[1].strip() for x in f]
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path, seed=None):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    cpnet_edges = list(cpnet.edges(data=True))
    if seed:
        random.seed(seed)
        random.shuffle(cpnet_edges)
    for u, v, data in cpnet_edges:
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


# def relational_graph_generation(qcs, acs, paths, rels):
#     raise NotImplementedError()  # TODO


# plain graph generation
# def plain_graph_generation(qcs, acs, paths, rels):
#     global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
#
#     graph = nx.Graph()
#     for p in paths:
#         for c_index in range(len(p) - 1):
#             h = p[c_index]
#             t = p[c_index + 1]
#             # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
#             graph.add_edge(h, t, weight=1.0)
#
#     for qc1, qc2 in list(itertools.combinations(qcs, 2)):
#         if cpnet_simple.has_edge(qc1, qc2):
#             graph.add_edge(qc1, qc2, weight=1.0)
#
#     for ac1, ac2 in list(itertools.combinations(acs, 2)):
#         if cpnet_simple.has_edge(ac1, ac2):
#             graph.add_edge(ac1, ac2, weight=1.0)
#
#     if len(qcs) == 0:
#         qcs.append(-1)
#
#     if len(acs) == 0:
#         acs.append(-1)
#
#     if len(paths) == 0:
#         for qc in qcs:
#             for ac in acs:
#                 graph.add_edge(qc, ac, rel=-1, weight=0.1)
#
#     g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
#     return nx.node_link_data(g)


# def generate_adj_matrix_per_inst(nxg_str):
#     global id2relation
#     n_rel = len(id2relation)
#
#     nxg = nx.node_link_graph(json.loads(nxg_str))
#     n_node = len(nxg.nodes)
#     cids = np.zeros(n_node, dtype=np.int32)
#     for node_id, node_attr in nxg.nodes(data=True):
#         cids[node_id] = node_attr['cid']
#
#     adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
#     for s in range(n_node):
#         for t in range(n_node):
#             s_c, t_c = cids[s], cids[t]
#             if cpnet_all.has_edge(s_c, t_c):
#                 for e_attr in cpnet_all[s_c][t_c].values():
#                     if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
#                         adj[e_attr['rel']][s][t] = 1
#     cids += 1
#     adj = coo_matrix(adj.reshape(-1, n_node))
#     return (adj, cids)


def concepts2adj(node_ids):
    """
    takes a list of concept IDs
    for all pairs, looks to see if there's an edge between them
    """
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s_c, t_c in all_pairs_edges(node_ids):
        s = np.where(cids == s_c)[0][0]
        t = np.where(cids == t_c)[0][0]
        for e_attr in cpnet[s_c][t_c].values():
            if 0 <= e_attr['rel'] < n_rel:
                adj[e_attr['rel']][s][t] = 1
    # cids += 1  # note!!! index 0 is reserved for padding
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids


def all_pairs_edges(nodes, unique=False):
    nodes = list(nodes)
    pairs = []
    for i, s in enumerate(nodes):
        if unique:
            iterator = nodes[i + 1:]
        else:
            iterator = nodes
        for t in iterator:
            if s == t:
                continue
            if cpnet.has_edge(s, t):
                pairs.append((s, t))
    return pairs


# def rels2adj(node_ids, rels, qid):
#     """
#     takes a list of concept IDs
#     for all pairs, looks to see if there's an edge between them
#     """
#     global id2relation
#     cids = np.array(node_ids, dtype=np.int32)
#     node_id_to_local = {node_id: i for i, node_id in enumerate(cids.tolist())}
#     n_rel = len(id2relation)
#     n_node = cids.shape[0]
#     adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
#     for s, r, t in rels:
#         assert cpnet.has_edge(s, t), f"No edge between ids {s} and {t} ({id2concept[s]} and {id2concept[t]}) QID {qid}"
#         adj[r][node_id_to_local[s]][node_id_to_local[t]] = 1
#
#     # cids += 1  # note!!! index 0 is reserved for padding
#     adj = coo_matrix(adj.reshape(-1, n_node))
#     return adj, cids


# def concepts_to_adj_matrices_1hop_neighbours(data):
#     qc_ids, ac_ids = data
#     qa_nodes = set(qc_ids) | set(ac_ids)
#     extra_nodes = set()
#     for u in set(qc_ids) | set(ac_ids):
#         if u in cpnet.nodes:
#             extra_nodes |= set(cpnet[u])
#     extra_nodes = extra_nodes - qa_nodes
#     schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
#     arange = np.arange(len(schema_graph))
#     qmask = arange < len(qc_ids)
#     amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
#     adj, concepts = concepts2adj(schema_graph)
#     return adj, concepts, qmask, amask


# def concepts_to_adj_matrices_1hop_neighbours_without_relatedto(data):
#     qc_ids, ac_ids = data
#     qa_nodes = set(qc_ids) | set(ac_ids)
#     extra_nodes = set()
#     for u in set(qc_ids) | set(ac_ids):
#         if u in cpnet.nodes:
#             for v in cpnet[u]:
#                 for data in cpnet[u][v].values():
#                     if data['rel'] not in (15, 32):
#                         extra_nodes.add(v)
#     extra_nodes = extra_nodes - qa_nodes
#     schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
#     arange = np.arange(len(schema_graph))
#     qmask = arange < len(qc_ids)
#     amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
#     adj, concepts = concepts2adj(schema_graph)
#     return adj, concepts, qmask, amask


# def concepts_to_adj_matrices_2hop_qa_pair(data):
#     qc_ids, ac_ids = data
#     qa_nodes = set(qc_ids) | set(ac_ids)
#     extra_nodes = set()
#     for qid in qc_ids:
#         for aid in ac_ids:
#             if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
#                 extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
#     extra_nodes = extra_nodes - qa_nodes
#     schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
#     arange = np.arange(len(schema_graph))
#     qmask = arange < len(qc_ids)
#     amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
#     adj, concepts = concepts2adj(schema_graph)
#     return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2hop_all_pair(max_node_num, data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                if len(extra_nodes) > max_node_num:
                    break
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def get_interim_nodes_via_pathfinding(qc_ids, ac_ids, k, max_node_num, max_eventual_edge_num):
    assert max_eventual_edge_num < 0, 'not implemented'
    extra_nodes = set()
    added_paths = 0
    ks = []
    for qid in qc_ids:
        if len(extra_nodes) > max_node_num > 0:
            break
        for aid in ac_ids:
            if qid == aid or qid not in cpnet_simple.nodes or aid not in cpnet_simple.nodes:
                continue
            try:
                for p in nx.shortest_simple_paths(cpnet_simple, source=qid, target=aid):
                    if (len(p) > k > 0) or (len(extra_nodes) > max_node_num > 0):
                        # if ((0 < k < len(p))
                        #         or (0 < max_node_num < len(extra_nodes))
                        #         or (0 < max_eventual_edge_num < len(all_pairs_edges(extra_nodes, True)))):
                        break
                    if len(p) >= 2:  # skip paths of length 1
                        # num_paths[len(p)] += 1
                        # nodes_at_path_len[len(p)] = nodes_at_path_len[len(p)].union(set(p))
                        extra_nodes |= set(p)
                        added_paths += 1
                        ks.append(len(p))
            except nx.NetworkXNoPath:
                continue
    return extra_nodes, ks


# def get_interim_nodes_via_pathfinding_max_overlap(max_node_num, data):
#     qc_ids, ac_ids, question = data
#     qa_nodes = set(qc_ids) | set(ac_ids)
#     edge_counter = Counter()
#
#     # TUNABLE
#     permissible_path_length_over_minimum = 2
#     max_paths_between_pair = 100
#
#     for qc, ac in itertools.product(qc_ids, ac_ids):
#         this_pair_shortest_paths_n = 0
#         if qc not in cpnet_simple.nodes or ac not in cpnet_simple.nodes:
#             continue
#         shortest_len = None
#         try:
#             for p in nx.shortest_simple_paths(cpnet_simple, source=qc, target=ac):
#                 if (shortest_len and len(p) > shortest_len + permissible_path_length_over_minimum) or this_pair_shortest_paths_n > max_paths_between_pair:
#                     break
#                 if not shortest_len:
#                     shortest_len = len(p)
#                 this_pair_shortest_paths_n += 1
#                 for pair in zip(p, p[1:]):
#                     edge_counter[tuple(sorted(pair))] += 1
#
#         except NetworkXNoPath:
#             # print("no path")
#             pass
#
#     G = nx.Graph()
#     for pair, count in edge_counter.items():
#         # 1 for a hop; reciprocal for how frequent it is (shortest path, so lower weight is better)
#         # TUNABLE
#         hop_weight = .5
#         G.add_edge(pair[0], pair[1], weight=hop_weight + (1 / count))
#
#     path_iterators = []
#     for qc, ac in itertools.product(qc_ids, ac_ids):
#         if qc not in G.nodes or ac not in G.nodes:
#             continue
#         try:
#             path_iterators.append(nx.shortest_simple_paths(G, source=qc, target=ac, weight='weight'))
#         except NetworkXNoPath:
#             continue
#
#     extra_nodes = set()
#     # First round - make sure all pairs are connected via the easiest path for them
#     for it in path_iterators:
#         # should be no error - NxNoPath caught above and won't be stopiteration on first try
#         extra_nodes |= set(next(it))
#
#     # Subsequently, take the lowest weighted paths in general across all iterators (just the top 10 from each)
#     subsequent_paths = []
#     # TUNABLE
#     top_n_paths_per_iterator = 10
#     for it in path_iterators:
#         try:
#             for _ in range(top_n_paths_per_iterator):
#                 subsequent_paths.append(next(it))
#         except StopIteration:
#             continue
#     subsequent_path_weights = [path_weight(G, path, 'weight') for path in subsequent_paths]
#
#     # And go through them from the lowest weighted first, til we hit our max number of nodes
#     paths_weights = sorted(zip(subsequent_paths, subsequent_path_weights), key=lambda x: x[0])
#     for p, _ in paths_weights:
#         potential_next = extra_nodes | set(p)
#         if len(potential_next) > max_node_num:
#             break
#         extra_nodes = potential_next
#
#     extra_nodes = extra_nodes - qa_nodes
#     # print(f"{extra_nodes} {len(qa_nodes) = } {len(extra_nodes) = }")
#     assert extra_nodes.intersection(set(qc_ids)) == set()
#     assert extra_nodes.intersection(set(ac_ids)) == set()
#     return sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes)


def concepts_to_adj_matrices_khop_all_pair(k, max_node_num, max_eventual_edge_num, data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    # num_paths = defaultdict(int)
    # nodes_at_path_len = defaultdict(set)
    extra_nodes, ks = get_interim_nodes_via_pathfinding(qc_ids, ac_ids, k, max_node_num, max_eventual_edge_num)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


# qc_ids, ac_ids, question = data
#     qa_nodes = set(qc_ids) | set(ac_ids)
#     extra_nodes = set()
#     for qid in qa_nodes:
#         for aid in qa_nodes:
#             if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
#                 extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
#     extra_nodes = extra_nodes - qa_nodes
#     return sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes)

def concepts_to_adj_matrices_khop_all_pair_qagnn_version(k, max_node_num, max_eventual_edge_num, data):
    qc_ids, ac_ids, question = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes, ks = get_interim_nodes_via_pathfinding(qc_ids, ac_ids, k, max_node_num, max_eventual_edge_num)
    extra_nodes = extra_nodes - qa_nodes
    assert extra_nodes.intersection(set(qc_ids)) == set()
    assert extra_nodes.intersection(set(ac_ids)) == set()
    return sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes), ks


uid_re = re.compile(r"[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}")
remove_punctuation_translation = str.maketrans('', '', string.punctuation)


# def find_expls_for_q_ans_OLD(sent, ans, question_text, explanations, require_ans_match=True):
#     # 'the sun appears to move across the sky each day rising in the east and setting in the west what causes this apparent motion'
#     expls = None
#     for q, e in zip(question_text, explanations):
#         if "? (A)" in q:
#             splat = q.split("? (A)")
#         else:
#             splat = q.split("(A)")
#         q_text = splat[0].lower().translate(remove_punctuation_translation).strip()
#
#         # print(f"q_text {q_text}")
#         # print(f"sent {sent}")
#         # should really match on question id, but the grounded q/a pairs don't have that
#         if q_text == sent or q_text in sent or sent in q_text:
#             # Some have this missing (bug)
#             if "(A)" in q and 'pad answer ' not in ans and require_ans_match:
#                 q_text_answers = q.split("(A)")[1].lower().translate(remove_punctuation_translation).strip()
#                 if ans in q_text_answers:
#                     if expls is not None:
#                         print(f"WARN already had a set of expls for {sent} {ans}")
#                     expls = uid_re.findall(e) if isinstance(e, str) else []
#             else:
#                 # it's ok if we get something wrong for the pad answer bit
#                 if expls is not None and 'pad answer ' in ans:
#                     continue
#                 if expls is not None:
#                     print(f"WARN already had a set of expls for {sent} {ans}")
#                 expls = uid_re.findall(e) if isinstance(e, str) else []
#     return expls


# def concepts_to_adj_matrices_wt_gold(id_to_gold_rels, method, data):
#     global concept2id, relation2id, cpnet_simple
#
#     qc_ids, ac_ids, sent, ans, q_id, is_correct = data
#
#     gold_rels = [[concept2id[s], relation2id[r], concept2id[o]] for s, r, o in id_to_gold_rels[q_id] if s != o]
#     all_gold_nodes = set(x[0] for x in gold_rels).union(set(x[2] for x in gold_rels))
#     all_nodes = set(all_gold_nodes)
#
#     paths_connecting_answer = []
#     if 'incorrect_cand_rels' in method and not is_correct and len(ac_ids.intersection(all_nodes)) == 0:
#         assert False, "this is wrongly implemented"
#         for gold_graph_node in all_gold_nodes:
#             for aid in ac_ids:
#                 for i, p in enumerate(nx.shortest_simple_paths(cpnet_simple, source=gold_graph_node, target=aid)):
#                     # max 10 paths between pairs of vertices
#                     if i > 10:
#                         break
#                     paths_connecting_answer.append(p)
#         paths_connecting_answer.sort(key=lambda x: len(x))
#         top_n_paths = 5
#         paths_connecting_answer = paths_connecting_answer[:top_n_paths]
#         additional_nodes = set(itertools.chain.from_iterable(paths_connecting_answer))
#         all_nodes = all_nodes.union(additional_nodes)
#
#     # if 'add_1_hop_neighbours' in method:
#     #     extra_nodes = set()
#     #     for gold_graph_node in all_gold_nodes:
#     #         extra_nodes |= set(x[1] for x in cpnet_simple.edges(gold_graph_node))
#     #     extra_nodes -= all_gold_nodes
#     #     # pct50 = int(len(all_gold_nodes) / 2)
#     #     # if len(extra_nodes) > pct50:
#     #         # extra_nodes = set(random.sample(extra_nodes, pct50))
#     #     all_nodes |= extra_nodes
#
#     if 'add_to_khop' in method:
#         k = 5
#         max_node_num = 200
#         nodes_from_khop = set()
#         # gold_rels = [[concept2id[s], relation2id[r], concept2id[o]] for s, r, o in id_to_gold_rels[q_id]]
#         for qid in qc_ids:
#             if len(nodes_from_khop) > max_node_num:
#                 break
#             for aid in ac_ids:
#                 if qid == aid or qid not in cpnet_simple.nodes or aid not in cpnet_simple.nodes or len(nodes_from_khop) > max_node_num:
#                     continue
#                 try:
#                     for p in nx.shortest_simple_paths(cpnet_simple, source=qid, target=aid):
#                         if len(p) > k or len(nodes_from_khop) > max_node_num:
#                             break
#                         if len(p) >= 2:  # skip paths of length 1
#                             for ss, oo in zip(p, p[1:]):
#                                 if cpnet[ss][oo][0]['rel'] < len(relation2id):
#                                     rr = cpnet[ss][oo][0]['rel']
#                                 else:
#                                     rr = cpnet[oo][ss][0]['rel']
#                                     ss, oo = oo, ss
#                                 paths_connecting_answer.append([ss, rr, oo])
#                             nodes_from_khop |= set(p)
#                 except nx.NetworkXNoPath:
#                     continue
#
#         nodes_from_khop -= all_gold_nodes
#         join_khop_to_gold = set()
#         for g in all_gold_nodes:
#             for e in nodes_from_khop:
#                 try:
#                     added = 0
#                     for i, p in enumerate(nx.shortest_simple_paths(cpnet_simple, source=g, target=e)):
#                         if len(p) > 3 or added == 2:
#                             break
#                         if len(p) >= 2:  # skip paths of length 1
#                             for ss, oo in zip(p, p[1:]):
#                                 if cpnet[ss][oo][0]['rel'] < len(relation2id):
#                                     rr = cpnet[ss][oo][0]['rel']
#                                 else:
#                                     rr = cpnet[oo][ss][0]['rel']
#                                     ss, oo = oo, ss
#                                 paths_connecting_answer.append([ss, rr, oo])
#                             join_khop_to_gold |= set(p)
#                             added += 1
#                 except nx.NetworkXNoPath:
#                     continue
#
#         all_nodes |= nodes_from_khop
#         all_nodes |= join_khop_to_gold
#
#     if 'add_all_rels' in method:
#         # adj, concepts = concepts2adj(schema_graph)
#         for c1 in all_nodes:
#             for c2 in all_nodes:
#                 if c1 == c2:
#                     continue
#                 if cpnet_simple.has_edge(c1, c2):
#                     if cpnet[c1][c2][0]['rel'] < len(relation2id):
#                         paths_connecting_answer.append((c1, cpnet[c1][c2][0]['rel'], c2))
#                     else:
#                         paths_connecting_answer.append((c2, cpnet[c2][c1][0]['rel'], c1))
#
#     if 'add_addtl_neighbours' in method and len(all_gold_nodes):
#         # i think this is nonsense but it's fine we don't use it
#         # surely this first loop does nothing
#         new_node_neighbours = defaultdict(set)
#         for c1 in all_gold_nodes:
#             for c2 in all_gold_nodes:
#                 if c1 == c2:
#                     continue
#                 try:
#                     for p in nx.shortest_simple_paths(cpnet_simple, source=c1, target=c2):
#                         if len(p) > 2:
#                             break
#                         if p[1] in all_gold_nodes:
#                             continue
#                         new_node_neighbours[p[1]].add(c1)
#                         new_node_neighbours[p[1]].add(c2)
#                 except nx.NetworkXNoPath:
#                     continue
#         new_node_neighbours_items = list(new_node_neighbours.items())
#         new_node_neighbours_items.sort(key=lambda x: -len(x[1]))
#         pct50 = len(all_gold_nodes) / 2
#         # print("addtl stage  2")
#         added_nodes = set()
#         for new_node, new_node_connections in new_node_neighbours_items:
#             if len(added_nodes) > pct50:
#                 break
#             added_nodes.add(new_node)
#             for connection in new_node_connections:
#                 if cpnet[new_node][connection][0]['rel'] < len(relation2id):
#                     paths_connecting_answer.append((new_node, cpnet[new_node][connection][0]['rel'], connection))
#                 else:
#                     paths_connecting_answer.append((connection, cpnet[connection][new_node][0]['rel'], new_node))
#         # print(f"addtl stage  3, {pct50} {len(added_nodes)}")
#         # for rest, just random
#         gold_nodes_list = list(all_gold_nodes)
#         new_nodes_per_old = [[y[1] for y in cpnet_simple.edges(n)] for n in gold_nodes_list]
#         while len(added_nodes) <= pct50:
#             for i, n in enumerate(new_nodes_per_old):
#                 g = gold_nodes_list[i]
#                 if len(n):
#                     p = n.pop()
#                     if p not in added_nodes:
#                         added_nodes.add(p)
#                         if cpnet[g][p][0]['rel'] < len(relation2id):
#                             paths_connecting_answer.append((g, cpnet[g][p][0]['rel'], p))
#                         else:
#                             paths_connecting_answer.append((p, cpnet[p][g][0]['rel'], g))
#
#         all_nodes |= added_nodes
#     qa_nodes = set(qc_ids) | set(ac_ids)
#     extra_nodes = all_nodes - qa_nodes
#
#     schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
#     arange = np.arange(len(schema_graph))
#     qmask = arange < len(qc_ids)
#     amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
#     adj, concepts = rels2adj(schema_graph, gold_rels + paths_connecting_answer, q_id)
#
#     return adj, concepts, qmask, amask


# def concepts_to_adj_matrices_wt_gold_OLD(ids, data):
#     global concept2id
#
#     questions_f = "/anfs/bigdisc/ga384/corpora/science_qa/WorldtreeExplanationCorpusV1_Sept2017_withMercury/tsv/questionsAndExplanations.tsv"
#     questions = pd.read_csv(questions_f, delimiter="\t")
#     qid, question_text, explanations = zip(*questions[['QuestionID', 'question', 'explanation']].values.tolist())
#     # explanations = [uid_re.findall(e) if isinstance(e, str) else [] for e in explanations]
#
#     qc_ids, ac_ids, sent, ans = data
#
#     assert sent.strip().endswith(ans), f"{sent} {ans}"
#     sent = sent.strip()[:-len(ans)].lower().translate(remove_punctuation_translation).strip()
#     ans = ans.lower().translate(remove_punctuation_translation).strip()
#     expls = find_expls_for_q_ans_OLD(sent, ans, question_text, explanations)
#     if expls is None:
#         expls = find_expls_for_q_ans_OLD(sent, ans, question_text, explanations, require_ans_match=False)
#
#     # TODO we are mostly there, deal with this later
#     # there are some missing explanations because the questions are 2019 worldtree but the explanation bank is from 2017
#     # if expls is None:
#     #     raise ValueError(f"No match found for sent '{sent}' '{ans}'")
#     if expls is None:
#         expls = []
#
#     expl_rows = [f"row-{f}" for f in expls]
#     row_concepts = set(concept2id[x] for x in expl_rows if x in concept2id)
#
#     qa_nodes = set(qc_ids) | set(ac_ids)
#     extra_nodes = set()
#
#     if not len(row_concepts):
#         # print("fallback")
#         # fallback
#         for qid in qa_nodes:
#             for aid in qa_nodes:
#                 if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
#                     extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
#         extra_nodes = extra_nodes - qa_nodes
#     else:
#         # print("no fallback")
#         for c in row_concepts:
#             extra_nodes |= set(cpnet_simple[c])
#         extra_nodes = row_concepts.union(extra_nodes)
#
#         qc_ids = qc_ids.intersection(extra_nodes)
#         ac_ids = ac_ids.intersection(extra_nodes)
#
#         extra_nodes = extra_nodes - qa_nodes
#
#     schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
#     arange = np.arange(len(schema_graph))
#     qmask = arange < len(qc_ids)
#     amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
#     adj, concepts = concepts2adj(schema_graph)
#
#     return adj, concepts, qmask, amask


# def concepts_to_adj_matrices_2step_relax_all_pair(data):
#     qc_ids, ac_ids = data
#     qa_nodes = set(qc_ids) | set(ac_ids)
#     extra_nodes = set()
#     for qid in qc_ids:
#         for aid in ac_ids:
#             if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
#                 extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
#     intermediate_ids = extra_nodes - qa_nodes
#     for qid in intermediate_ids:
#         for aid in ac_ids:
#             if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
#                 extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
#     for qid in qc_ids:
#         for aid in intermediate_ids:
#             if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
#                 extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
#     extra_nodes = extra_nodes - qa_nodes
#     schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
#     arange = np.arange(len(schema_graph))
#     qmask = arange < len(qc_ids)
#     amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
#     adj, concepts = concepts2adj(schema_graph)
#     return adj, concepts, qmask, amask


# def concepts_to_adj_matrices_3hop_qa_pair(data):
#     qc_ids, ac_ids = data
#     qa_nodes = set(qc_ids) | set(ac_ids)
#     extra_nodes = set()
#     for qid in qc_ids:
#         for aid in ac_ids:
#             if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
#                 for u in cpnet_simple[qid]:
#                     for v in cpnet_simple[aid]:
#                         if cpnet_simple.has_edge(u, v):  # ac is a 3-hop neighbour of qc
#                             extra_nodes.add(u)
#                             extra_nodes.add(v)
#                         if u == v:  # ac is a 2-hop neighbour of qc
#                             extra_nodes.add(u)
#     extra_nodes = extra_nodes - qa_nodes
#     schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
#     arange = np.arange(len(schema_graph))
#     qmask = arange < len(qc_ids)
#     amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
#     adj, concepts = concepts2adj(schema_graph)
#     return adj, concepts, qmask, amask


######################################################################


class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, masked_lm_labels=None):
        assert attention_mask is not None
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        # hidden_states of final layer (batch_size, sequence_length, hidden_size)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                      masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs


TOKENIZER = None
LM_MODEL = None


def get_lm_score(cids, question, id2concept):
    global TOKENIZER, LM_MODEL
    if TOKENIZER is None:
        logger.info('loading pre-trained LM...')
        TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
        LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
        LM_MODEL.cuda()
        LM_MODEL.eval()
        logger.info('loading done')

    cids = cids[:]
    # QAcontext node
    cids.insert(0, -1)
    sents, scores = [], []
    for cid in cids:
        if cid == -1:
            sent = question.lower()
        else:
            sent = '{} {}.'.format(question.lower(), ' '.join(id2concept[cid].split('_')))
        sent = TOKENIZER.encode(sent, add_special_tokens=True)
        sents.append(sent)
    n_cids = len(cids)
    cur_idx = 0
    batch_size = 256
    while cur_idx < n_cids:
        # Prepare batch
        input_ids = sents[cur_idx: cur_idx + batch_size]
        max_len = max([len(seq) for seq in input_ids])
        for j, seq in enumerate(input_ids):
            seq += [TOKENIZER.pad_token_id] * (max_len - len(seq))
            input_ids[j] = seq
        # [B, seq len]
        input_ids = torch.tensor(input_ids).cuda()
        # [B, seq_len]
        mask = (input_ids != 1).long()

        # Get LM score
        with torch.no_grad():
            outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
            # [B, ]
            loss = outputs[0]
            # list of float
            _scores = list(-loss.detach().cpu().numpy())
        scores += _scores
        cur_idx += batch_size
    assert len(sents) == len(scores) == len(cids)
    cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1]))  # score: from high to low
    return cid2score


def concepts_to_adj_matrices_2hop_all_pair__use_lm__part1(data):
    qc_ids, ac_ids, question = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    return sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes)


def concepts_to_adj_matrices_2hop_all_pair__use_lm__part2(data):
    global id2concept
    qc_ids, ac_ids, question, extra_nodes = data
    cid2score = get_lm_score(qc_ids + ac_ids + extra_nodes, question, id2concept)
    return qc_ids, ac_ids, question, extra_nodes, cid2score


def concepts_to_adj_matrices_2hop_all_pair__use_lm__part3(data):
    qc_ids, ac_ids, question, extra_nodes, cid2score = data
    schema_graph = qc_ids + ac_ids + sorted(extra_nodes, key=lambda x: -cid2score[x])  # score: from high to low
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {'adj': adj, 'concepts': concepts, 'qmask': qmask, 'amask': amask, 'cid2score': cid2score}


################################################################################


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


# def generate_graph(grounded_path, pruned_paths_path, cpnet_vocab_path, cpnet_graph_path, output_path):
#     print(f'generating schema graphs for {grounded_path} and {pruned_paths_path}...')
#
#     global concept2id, id2concept, relation2id, id2relation
#     if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
#         load_resources(cpnet_vocab_path)
#
#     global cpnet, cpnet_simple
#     if cpnet is None or cpnet_simple is None:
#         load_cpnet(cpnet_graph_path)
#
#     nrow = sum(1 for _ in open(grounded_path, 'r'))
#     with open(grounded_path, 'r') as fin_gr, \
#             open(pruned_paths_path, 'r') as fin_pf, \
#             open(output_path, 'w') as fout:
#         for line_gr, line_pf in tqdm(zip(fin_gr, fin_pf), total=nrow):
#             mcp = json.loads(line_gr)
#             qa_pairs = json.loads(line_pf)
#
#             statement_paths = []
#             statement_rel_list = []
#             for qas in qa_pairs:
#                 if qas["pf_res"] is None:
#                     cur_paths = []
#                     cur_rels = []
#                 else:
#                     cur_paths = [item["path"] for item in qas["pf_res"]]
#                     cur_rels = [item["rel"] for item in qas["pf_res"]]
#                 statement_paths.extend(cur_paths)
#                 statement_rel_list.extend(cur_rels)
#
#             qcs = [concept2id[c] for c in mcp["qc"]]
#             acs = [concept2id[c] for c in mcp["ac"]]
#
#             gobj = plain_graph_generation(qcs=qcs, acs=acs,
#                                           paths=statement_paths,
#                                           rels=statement_rel_list)
#             fout.write(json.dumps(gobj) + '\n')
#
#     print(f'schema graphs saved to {output_path}')
#     print()


# def generate_adj_matrices(ori_schema_graph_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes, num_rels=34, debug=False):
#     print(f'generating adjacency matrices for {ori_schema_graph_path} and {cpnet_graph_path}...')
#
#     global concept2id, id2concept, relation2id, id2relation
#     if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
#         load_resources(cpnet_vocab_path)
#
#     global cpnet_all
#     if cpnet_all is None:
#         cpnet_all = nx.read_gpickle(cpnet_graph_path)
#
#     with open(ori_schema_graph_path, 'r') as fin:
#         nxg_strs = [line for line in fin]
#
#     if debug:
#         nxgs = nxgs[:1]
#
#     with Pool(num_processes) as p:
#         res = list(tqdm(p.imap(generate_adj_matrix_per_inst, nxg_strs), total=len(nxg_strs)))
#
#     with open(output_path, 'wb') as fout:
#         pickle.dump(res, fout)
#
#     print(f'adjacency matrices saved to {output_path}')
#     print()


def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path,
                                             output_path, num_processes, graph, k=-1,
                                             max_node_num=200,
                                             max_eventual_edge_num=-1, seed=None):
    if os.path.exists(output_path):
        print(f'adj data for {grounded_path} already exists')
        return
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format
    one for each (Q, a_cand) pair
    
    NOTE this function will save data in a different format to generate_adj_data_from_grounded_concepts__use_lm.
    This will save a list of tuples, where each tuple is (adj, concepts, qmask, amask).
    That will save a list of dictionaries, and also include relevancy scores per concept (cid2score).
    MHGRN can be run with either format, but QA-GNN requires the dictionary format, because it needs the LM scores.
    You can add LM scores to outputs generated by this function by running add_lm_score_to_graph.py
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path, graph)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path, seed=seed)

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids))

    with Pool(num_processes) as p:
        chosen_function = partial(concepts_to_adj_matrices_khop_all_pair, k, max_node_num, max_eventual_edge_num)
        res = list(tqdm(p.imap(chosen_function, qa_data), total=len(qa_data)))

    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()


def generate_adj_data_from_grounded_concepts__use_lm(grounded_path, cpnet_graph_path, cpnet_vocab_path,
                                                     output_path, num_processes, graph, k=-1,
                                                     max_node_num=200, max_eventual_edge_num=-1, seed=None):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
        (5) cid2score that maps a concept id to its relevance score given the QA context
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    if os.path.exists(output_path):
        print(f'adj data for {grounded_path} already exists')
        return
    print(f'generating adj data (with LM) for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path, graph)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path, seed=seed)

    qa_data = []
    statement_path = grounded_path.replace('grounded', 'statement')
    with (open(grounded_path, 'r', encoding='utf-8') as fin_ground,
          open(statement_path, 'r', encoding='utf-8') as fin_state):
        lines_ground = fin_ground.readlines()
        lines_state = fin_state.readlines()
        assert len(lines_ground) % len(lines_state) == 0
        n_choices = len(lines_ground) // len(lines_state)
        for j, line in enumerate(lines_ground):
            dic = json.loads(line)
            q_ids = set(concept2id[c.replace("_", " ")] for c in dic['qc'])
            a_ids = set(concept2id[c.replace("_", " ")] for c in dic['ac'])
            q_ids = q_ids - a_ids
            statement_obj = json.loads(lines_state[j // n_choices])
            qa_context = "{} {}.".format(statement_obj['question']['stem'], dic['ans'])
            qa_data.append((q_ids, a_ids, qa_context))

    with Pool(num_processes) as p:
        chosen_function = partial(concepts_to_adj_matrices_khop_all_pair_qagnn_version, k,
                                  max_node_num, max_eventual_edge_num)
        res1 = list(tqdm(p.imap(chosen_function, qa_data), total=len(qa_data), desc=f"Part 1 - {k} hop"))

        *res1, ks = zip(*res1)
        res1 = zip(*res1)
        ks = list(itertools.chain.from_iterable(ks))
        print("path length mean", np.mean(ks), "std", np.std(ks))

    res2 = []
    for j, _data in tqdm(list(enumerate(res1)), desc="Getting LM scores"):
        res2.append(concepts_to_adj_matrices_2hop_all_pair__use_lm__part2(_data))

    with Pool(num_processes) as p:
        res3 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_lm__part3, res2),
                         total=len(res2), desc="Part 3"))

    # res is a list of responses
    with open(output_path, 'wb') as fout:
        pickle.dump(res3, fout)

    print(f'adj data saved to {output_path}')
    print()
