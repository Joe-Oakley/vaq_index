import sys
sys.path.append('src')
from pathlib import Path

from dataset import DataSet
from transformed import TransformedDataSet
from vaqindex import VAQIndex
from queryset import QuerySet

def calc_valid_num_blocks(num_vectors):
    valids = []
    for i in range(20):
        if num_vectors % i == 0:
            valids.append(i)
    print("Valid block counts for " , str(num_vectors) , " vectors: ", str(valids))

    return valids

def check_block_count_validity(num_vectors, num_blocks):
    if num_vectors % num_blocks == 0:
        print("Valid number of blocks selected: ", str(num_blocks))
        return True
    else:
        print("Invalid number of blocks selected.", str(num_blocks))
        return False

def main():

    # path = Path('datasets/histo64i64_12103/')
    # fname = 'histo64i64_12103_swapped'
    # num_vectors = 12103
    # num_dimensions = 64
    # num_blocks = 7
    
    # path = Path('datasets/siftsmall/')
    # fname = 'siftsmall'
    # num_vectors = 10000
    # num_dimensions = 128
    # num_blocks = 10
    
    # path = Path('datasets/ltest/')    
    # fname = 'ltest'
    # num_vectors = 50
    # num_dimensions = 128
    # num_blocks = 1

    path = Path('datasets/sift1m/')
    fname = 'sift1m'
    num_vectors = 1000000
    num_dimensions = 128
    num_blocks = 10

    print("Checking num_blocks validity")
    if not check_block_count_validity(num_vectors, num_blocks):
        exit(1)

    dataset = DataSet(path, fname, num_vectors, num_dimensions, num_blocks, big_endian=False)
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