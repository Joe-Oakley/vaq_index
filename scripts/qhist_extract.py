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
        self.word_size = word_size # 4 bytes default (float32)
        self.num_words_per_block = 0
        self.num_vectors_per_block = 0        
        self.file_handle_in = None
        self.file_handle_out = None
        self.sample_inds = None
        self.full_qhist_name = None

    #----------------------------------------------------------------------------------------------------------------------------------------    
    def _open_file(self, type):
        if type == 'in':
           self.file_handle_in = open(self.full_fname_in, mode="rb")
        elif type == 'out':
           self.file_handle_out = open(self.full_fname_out, mode="wb")            
        else:
            assert ValueError("Invalid File Handle Type ", type, ' passed to _open_file')
            
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self, type):
        
        if type == 'in':
           self.file_handle_in.close()
        elif type == 'out':
           self.file_handle_out.close()
        else:
            assert ValueError("Invalid File Handle Type ", type, ' passed to _close_file')
        
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _save_ids(self):
        np.savez(self.full_fname_out_ids, IDS=np.array(self.sample_inds))
        self._load_ids()

    #----------------------------------------------------------------------------------------------------------------------------------------   
    def _load_ids(self):

        with np.load(self.full_fname_out_ids + ".npz") as data:
            ids = data['IDS']
            print()
            print("Verifying saved ids : ")
            print(ids)
            print()

    #---------------------------------------------------------------------------------------------------------------------------------------- 
    def _initialise(self):

        np.set_printoptions(suppress=True)

        self.full_fname_in = os.path.join(self.path, '') + self.fname
        self.full_qhist_name = self.full_fname_in + ".qhist"        
        self.full_fname_out = os.path.join(self.path, '') + self.fname + "_qry"

        self.full_fname_out_ids = self.full_fname_out 
        self.sample_inds = []
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _read_vector(self, vector_idx):

        # offset = int(0)
        # vector_idx is the index of the required vector, assuming zero counting
        offset = np.int64( np.int64(vector_idx) * np.int64(self.num_dimensions + 1) * np.int64(self.word_size) )

        self.file_handle_in.seek(offset, os.SEEK_SET)
        vector = np.fromfile(file=self.file_handle_in, count=(self.num_dimensions + 1), dtype=np.float32)

        return vector

    #----------------------------------------------------------------------------------------------------------------------------------------
    def analyse_qhist(self):

        with open(self.full_qhist_name, mode='rb') as f:

            byte_counter = 0

            # Query History file may contain outputs from multiple query runs. Process until end of file reached.
            while f.tell() < os.fstat(f.fileno()).st_size:

                f.seek(byte_counter, os.SEEK_SET) # Seek takes a byte counter
                num_queries_qhist = np.fromfile(file=f, count=1, dtype=np.uint32)[0] # Count is in words
                byte_counter += self.word_size

                for i in range(num_queries_qhist):

                    # Read 4 words (np.fromfile): query_id, query_k, phase 1 elims, phase 2 visits
                    f.seek(byte_counter, os.SEEK_SET)
                    q_info = np.fromfile(file=f, count = 4, dtype=np.uint32)

                    q_info = np.reshape(q_info, (4,1), order="C")
                    q_id, q_k, q_p1, q_p2 = q_info[0,0], q_info[1,0], q_info[2,0], q_info[3,0] 
                    byte_counter += 4 * self.word_size

                    # Read next (2*query_k) words (np.fromfile) -> reshape into (k, 2) distances matrix, probably order C
                    f.seek(byte_counter, os.SEEK_SET)
                    distances = np.fromfile(file=f, count = 2*q_k, dtype=np.float32)
                    distances = np.reshape(distances, (q_k, 2), order='C')
                    byte_counter += 2 * q_k * self.word_size
                       
                    # Loop over rows k of distances matrix
                    for j in range(q_k):

                        # Update self.weights[k]
                        vector_id = int(distances[j,0])
                        self.sample_inds.append(vector_id)
                        
        self.sample_inds.sort()

        print()
        print('Random sample of vectors : Selecting:')
        print(self.sample_inds)
        self._save_ids()

    #----------------------------------------------------------------------------------------------------------------------------------------
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
        
    #----------------------------------------------------------------------------------------------------------------------------------------
    def process(self):
        
        # Initialisations
        self._initialise()
        
        self.analyse_qhist()

        self.build_samples_file()

    #----------------------------------------------------------------------------------------------------------------------------------------

def main():

    # path = Path('datasets/histo64i64_12103/')    
    # fname = 'histo64i64_12103_swapped'
    # num_vectors = 12103
    # num_dimensions = 64
    # num_blocks = 7 
    # samples = 12

    path = Path('datasets/siftsmall/')
    fname = 'siftsmall'    
    num_vectors = 10000
    num_dimensions = 128
    num_samples = 1
    
    # path = Path('datasets/sift1m/')
    # fname = 'sift1m'    
    # num_vectors = 1000000
    # num_dimensions = 128
    # samples = 10
    
    # path = Path('datasets/gist1m/')
    # fname = 'gist1m'    
    # num_vectors = 1000000
    # num_dimensions = 960
    # samples = 10
    
    dataset = DataSet(path, fname, num_vectors, num_dimensions, num_samples)
    dataset.process()


if __name__ == "__main__":
    main()