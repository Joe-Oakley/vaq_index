import sys

sys.path.append('src')
from pathlib import Path
import argparse
from dataset import DataSet
from transformed import TransformedDataSet
from vaqindex import VAQIndex
from queryset import QuerySet


def calc_valid_num_blocks(num_vectors):
    valids = []
    for i in range(20):
        if num_vectors % i == 0:
            valids.append(i)
    print("Valid block counts for ", str(num_vectors), " vectors: ", str(valids))

    return valids


def check_block_count_validity(num_vectors, num_blocks):
    if num_vectors % num_blocks == 0:
        print("Valid number of blocks selected: ", str(num_blocks))
        return True
    else:
        print("Invalid number of blocks selected.", str(num_blocks))
        return False


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fname", type=str, required=True)
    parser.add_argument("-num_vectors", type=int, required=True)
    parser.add_argument("-num_dimensions", type=int, required=True)
    parser.add_argument("-num_blocks", type=int, required=True)
    parser.add_argument("-big_endian", type=bool, default=False)
    parser.add_argument("-dsmode", type=str, choices=("B",), default="B")
    return parser.parse_args()


def main():
    args = parse()
    path = Path(f"./datasets/{args.fname}")
    print("Checking num_blocks validity")
    if not check_block_count_validity(args.num_vectors, args.num_blocks):
        exit(1)

    dataset = DataSet(path, args.fname, args.num_vectors, args.num_dimensions, args.num_blocks, big_endian=args.big_endian, dsmode=args.dsmode)
    dataset.process()
    print()
    print("************************************************************")
    print("Finished processing in dataset.py! Starting transformed.py!")
    print("************************************************************")
    print()

    tf_dataset = TransformedDataSet(path, 'B', DS=dataset)
    tf_dataset.build()
    print()
    print("************************************************************")
    print("Finished processing in transformed.py! Starting vaqindex.py!")
    print("************************************************************")
    print()

    vaq_index = VAQIndex(q_lambda=1, vaqmode='B', bit_budget=500, tf_dataset=tf_dataset)
    vaq_index.build()
    print()
    print("************************************************************")
    print("Finished processing in vaqindex.py! Starting queryset.py!")
    print("************************************************************")
    print()

    queryset = QuerySet(query_k=5, vaqindex=vaq_index)
    # queryset = QuerySet(query_k=100, vaqindex=vaq_index)
    queryset.process()


if __name__ == "__main__":
    main()
