import sys
sys.path.append('src')
from pathlib import Path

from dataset import DataSet
from transformed import TransformedDataSet
from vaqindex import VAQIndex
from queryset import QuerySet

def main():

    # path = Path('datasets/histo64i64_12103/')
    # fname = 'histo64i64_12103_swapped'
    # num_vectors = 12103
    # num_dimensions = 64
    # num_blocks = 7
    
    path = Path('datasets/siftsmall/')
    fname = 'siftsmall'
    num_vectors = 10000
    num_dimensions = 128
    num_blocks = 10
    
    # path = Path('datasets/ltest/')    
    # fname = 'ltest'
    # num_vectors = 100
    # num_dimensions = 128
    # num_blocks = 2

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

    queryset = QuerySet(query_k=2, vaqindex=vaq_index)
    queryset.process()


    
if __name__ == "__main__":
    main()