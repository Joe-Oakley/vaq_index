import sys

sys.path.append('src')
from pathlib import Path
import numpy as np
from numpy import linalg as LA
import os
import random


class DataSet:
    def __init__(self, path, fname, num_vectors=None, num_dimensions=None, num_samples=None, word_size=4):
        self.path = path
        self.fname = fname
        self.full_fname_in = None
        self.full_fname_out = None
        self.full_fname_out_ids = None
        self.num_vectors = num_vectors
        self.num_dimensions = num_dimensions
        self.num_samples = num_samples
        self.word_size = word_size  # 4 bytes default (float32)
        self.num_words_per_block = 0
        self.num_vectors_per_block = 0
        self.file_handle_in = None
        self.file_handle_out = None
        self.sample_inds = None

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _open_file(self, type):
        if type == 'in':
            self.file_handle_in = open(self.full_fname_in, mode="rb")
        elif type == 'out':
            self.file_handle_out = open(self.full_fname_out, mode="wb")
        else:
            assert ValueError("Invalid File Handle Type ", type, ' passed to _open_file')

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self, type):

        if type == 'in':
            self.file_handle_in.close()
        elif type == 'out':
            self.file_handle_out.close()
        else:
            assert ValueError("Invalid File Handle Type ", type, ' passed to _close_file')

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _save_ids(self):
        np.savez(self.full_fname_out_ids, IDS=np.array(self.sample_inds))
        self._load_ids()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_ids(self):

        with np.load(self.full_fname_out_ids + ".npz") as data:
            ids = data['IDS']
            print()
            print("Verifying saved ids : ")
            print(ids)
            print()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialise(self):

        np.set_printoptions(suppress=True)

        # total_file_words = (self.num_vectors * (self.num_dimensions+1))
        # assert total_file_words % self.num_blocks == 0
        # assert (total_file_words / self.num_blocks) % (self.num_dimensions + 1) == 0, "Inconsistent number of blocks selected."

        self.full_fname_in = os.path.join(self.path, '') + self.fname
        self.full_fname_out = os.path.join(self.path, '') + self.fname + "_qry"
        self.full_fname_out_ids = self.full_fname_out

        # #----------------------------------------------------------------------------------------------------------------------------------------

    # def _read_vector(self, vector_idx):

    #     offset = 0
    #     # vector_idx is the index of the required vector, assuming zero counting
    #     offset = vector_idx * (self.num_dimensions + 1) * self.word_size

    #     self.file_handle_in.seek(offset, os.SEEK_SET)
    #     vector = np.fromfile(file=self.file_handle_in, count=(self.num_dimensions + 1), dtype=np.float32)

    #     return vector

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _read_vector(self, vector_idx):

        # offset = int(0)
        # vector_idx is the index of the required vector, assuming zero counting
        offset = np.int64(np.int64(vector_idx) * np.int64(self.num_dimensions + 1) * np.int64(self.word_size))

        self.file_handle_in.seek(offset, os.SEEK_SET)
        vector = np.fromfile(file=self.file_handle_in, count=(self.num_dimensions + 1), dtype=np.float32)

        return vector

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def generate_sample_inds(self):

        # all_vector_inds = np.arange(0,self.num_vectors)
        all_vector_inds = list(np.arange(0, self.num_vectors, dtype=np.int32))

        # self.sample_inds = random.sample(all_vector_inds, self.num_samples)
        sel = random.sample(all_vector_inds, self.num_samples)
        sel.sort()
        self.sample_inds = sel

        print()
        print("Random sample of vectors : Selecting:")
        print(self.sample_inds)
        # print(type(self.sample_inds))
        print()

        self._save_ids()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def build_samples_file(self):

        # Open Input/Output files
        self._open_file('in')
        self._open_file('out')

        read_count = 0
        write_count = 0

        for ind in self.sample_inds:
            vec = self._read_vector(ind)
            read_count += 1
            self.file_handle_out.write(vec)
            write_count += 1

        print()
        print("Vectors Read    : ", str(read_count))
        print("Vectors Written : ", str(write_count))
        print()

        # Close file
        self._close_file('in')
        self._close_file('out')

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def process(self):

        # Initialisations
        self._initialise()

        self.generate_sample_inds()

        self.build_samples_file()

    # ----------------------------------------------------------------------------------------------------------------------------------------


def main():
    # path = Path('datasets/histo64i64_12103/')
    # fname = 'histo64i64_12103_swapped'
    # num_vectors = 12103
    # num_dimensions = 64
    # num_blocks = 7 
    # samples = 12

    # path = Path('datasets/siftsmall/')
    # fname = 'siftsmall'    
    # num_vectors = 10000
    # num_dimensions = 128
    # num_blocks = 1 
    # samples = 10 

    # path = Path('datasets/sift1m/')
    # fname = 'sift1m'    
    # num_vectors = 1000000
    # num_dimensions = 128
    # samples = 10

    path = Path('datasets/gist1m/')
    fname = 'gist1m'
    num_vectors = 1000000
    num_dimensions = 960
    samples = 10

    dataset = DataSet(path, fname, num_vectors, num_dimensions, samples)
    dataset.process()


if __name__ == "__main__":
    main()
