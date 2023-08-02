import argparse
import pickle
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    for split in ['train', 'dev', 'test']:
        file = f'{args.dir}/{split}.graph.adj.pk'

        with open(file, 'rb') as f:
            data = pickle.load(f)

        random.seed(args.seed)
        random.shuffle(data)

        newfile = file.replace(".graph.adj.pk", "") + ".swapped" + ".graph.adj.pk"
        with open(newfile, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    main()
