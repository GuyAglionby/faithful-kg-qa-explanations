import argparse
import glob
import os
import pickle
import numpy as np

import jsonlines
from scipy.sparse import coo_matrix
from tqdm import tqdm

from utils.graph import get_lm_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs='+')
    args = parser.parse_args()

    files = []
    id2concepts = []
    for d in args.dirs:
        this_dir_files = list(glob.glob(f"{d}/graph/*.adj.pk"))
        files.extend(this_dir_files)
        if d.endswith("/"):
            d = d[:-1]
        tld = d.split("/")[-1]
        dataset = tld.split('-')[0]
        vocab_file = "/".join(d.split("/")[:-1]) + f"/{tld[len(dataset) + 1:]}/concept.txt"
        with open(vocab_file) as f:
            this_dir_vocab = [x.strip() for x in f]
            for _ in range(len(this_dir_files)):
                id2concepts.append(this_dir_vocab)

    for filename, id2concept in tqdm(zip(files, id2concepts), desc="going through graphs", total=len(files)):
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)
        if isinstance(save_data[0], dict):
            continue
        assert isinstance(save_data[0], tuple)

        statement_filename = filename.replace('.graph.adj.pk', '.grounded.jsonl').replace("/graph/", "/grounded/")
        with jsonlines.open(statement_filename) as f:
            grounded = [obj['sent'] for obj in f]

        data = [{'adj': x[0], 'concepts': x[1], 'qmask': x[2], 'amask': x[3], 'sent': sent}
                for x, sent in zip(save_data, grounded)]

        new_file_out = []
        for j, d in tqdm(enumerate(data), desc="going through individual questions", total=len(data)):
            # For each concept, get the score of how much it relates to `sent` (which is a a (q;a) instance)
            concepts = d['concepts'].tolist()
            sent = d['sent']
            cid2score = get_lm_score(concepts, sent, id2concept)

            # Quick sanity check that the ordering of the concepts in the loaded data is as we expect
            # (q concepts, a concepts, other concepts)
            # (I think) we rely on this later, so even though it basically is guaranteed by the saving mechanism check anyway
            other_mask = ~(d['qmask'] | d['amask'])
            q_idxs = np.where(d['qmask'])[0].tolist()
            a_idxs = np.where(d['amask'])[0].tolist()
            other_idxs = np.where(other_mask)[0].tolist()

            try:
                assert max(q_idxs) < min(a_idxs)
            except ValueError:
                pass
            except AssertionError as e:
                print("something is wrong with question/answer concept ordering in this graph")
                raise e
            try:
                assert max(a_idxs) < min(other_idxs)
            except ValueError:
                pass
            except AssertionError as e:
                print("something is wrong with question/answer concept ordering in this graph")
                raise e

            # Translate the old adj tensor to parallel lists, and make those (concept) lists have global idx not local
            old_adj = d['adj'].toarray().reshape((-1, len(concepts), len(concepts)))
            rel_id, s_id_old, o_id_old = old_adj.nonzero()
            s_id_old = d['concepts'][s_id_old].tolist()
            o_id_old = d['concepts'][o_id_old].tolist()

            # Build schema graph
            qc_ids = d['concepts'][d['qmask']].tolist()
            ac_ids = d['concepts'][d['amask']].tolist()
            extra_nodes = d['concepts'][other_mask].tolist()

            schema_graph = qc_ids + ac_ids + sorted(extra_nodes, key=lambda x: -cid2score[x])
            schema_graph_id_to_i = {id_: i for i, id_ in enumerate(schema_graph)}
            arange = np.arange(len(schema_graph))
            qmask = arange < len(qc_ids)
            amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))

            # Build new adj, using mapping to new concept order
            adj_new = np.zeros(old_adj.shape, dtype=np.uint8)
            for r, s, o in zip(rel_id, s_id_old, o_id_old):
                adj_new[r][schema_graph_id_to_i[s]][schema_graph_id_to_i[o]] = 1

            concepts = np.array(schema_graph, dtype=np.int32)
            adj_new = coo_matrix(adj_new.reshape(-1, len(schema_graph)))
            new_file_out.append({'adj': adj_new, 'concepts': concepts,
                                 'qmask': qmask, 'amask': amask, 'cid2score': cid2score})
        # else:
        #     data['cid2score'] = cid2score
        #     new_file_out.append(data)
        # adj, concepts = concepts2adj(schema_graph)

        os.rename(filename, filename + "_NO_CID2SCORE")
        with open(filename, 'wb') as fout:
            pickle.dump(new_file_out, fout)


if __name__ == '__main__':
    main()