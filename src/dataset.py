import numpy as np
from numpy import linalg as LA
import os
import json

class DataSet:
    def __init__(self, path, fname, num_vectors=None, num_dimensions=None, num_blocks=None, word_size=4, big_endian=False, dsmode='B'):
        self.path = path
        self.fname = fname
        self.full_fname = None
        self.num_vectors = num_vectors
        self.num_dimensions = num_dimensions
        self.num_blocks = num_blocks
        self.word_size = word_size # 4 bytes default (float32)
        self.big_endian = big_endian
        self.dsmode = dsmode

        self.num_words_per_block = 0
        self.num_vectors_per_block = 0
        self.file_handle = None
        self.dim_means = None
        self.cov_matrix = None
        self.transform_matrix = None
        self.param_dict = None
    #----------------------------------------------------------------------------------------------------------------------------------------    
    def _open_file(self):
        self.file_handle = open(self.full_fname, mode="rb")
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self):
        self.file_handle.close()
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _initialise(self):
        # An item is a 4 byte chunk. Each vector will have (num_dimensions+1) items. The first is a 
        # record descriptor (ID), followed by 1 item per dimension.

        total_file_words = (self.num_vectors * (self.num_dimensions+1))
        assert (total_file_words / self.num_blocks) % (self.num_dimensions + 1) == 0, "Inconsistent number of blocks selected."
        self.num_words_per_block = int(total_file_words / self.num_blocks)
        self.num_vectors_per_block = int(self.num_words_per_block / (self.num_dimensions + 1))

        np.set_printoptions(suppress=True)
    
        self.full_fname = os.path.join(self.path, '') + self.fname
        self.dim_means = np.zeros((1, self.num_dimensions), dtype=np.float32)
        self.cov_matrix = np.zeros((self.num_dimensions, self.num_dimensions), dtype=np.float32)
        self.transform_matrix = np.zeros((self.num_dimensions, self.num_dimensions), dtype=np.float32)
         
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _read_block(self, block_idx):

        #(num_dims+1, block_size/(num_dims+1)) -> (num_dims, num_vectors_per_block) if all as float32

        self.file_handle.seek(self.num_words_per_block*block_idx*self.word_size, os.SEEK_SET) # Multiply by word_size since seek wants a byte location.
        if self.big_endian:
            block = np.fromfile(file=self.file_handle, count=self.num_words_per_block, dtype=np.float32).byteswap(inplace=True)
        else:
            block = np.fromfile(file=self.file_handle, count=self.num_words_per_block, dtype=np.float32)

        block = np.reshape(block, (self.num_vectors_per_block, self.num_dimensions+1), order="C")
        block = np.delete(block, 0, 1)

        return block
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    def generate_dataset_block(self,start_offset=0):

        block_idx = start_offset
        with open(self.full_fname, mode="rb") as f:
            
            while True:
                f.seek(self.num_words_per_block*block_idx*self.word_size, os.SEEK_SET) # Multiply by word_size since seek wants a byte location.
                if self.big_endian:
                    block = np.fromfile(file=f, count=self.num_words_per_block, dtype=np.float32).byteswap(inplace=True)
                else:
                    block = np.fromfile(file=f, count=self.num_words_per_block, dtype=np.float32)

                if block.size > 0:
                    block = np.reshape(block, (self.num_vectors_per_block, self.num_dimensions+1), order="C")
                    block = np.delete(block, 0, 1)
                    # AT THIS POINT, EACH COL IS A VECTOR AND EACH ROW IS AN ATTRIBUTE (FOR 1/NUM_BLOCKS OF THE VECTORS)

                    yield block
                    block_idx +=1
                else:
                    break

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _calc_dim_means(self):

        self._open_file()
        # Loop over number of blocks
        for i in range(self.num_blocks):
            block = self._read_block(i)
            dim_sums = np.sum(block,axis=0).reshape(1, self.num_dimensions)  # (1,128)
            self.dim_means = np.add(self.dim_means, dim_sums)
       
        self._close_file()
        self.dim_means = np.float32(self.dim_means / self.num_vectors)

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _calc_cov_matrix(self):
        # self.cov_matrix is (num_dimensions, num_dimensions)

        # Open file
        self._open_file()

        # Generate repmat(dim_means, 1, num_vectors_per_block)
        rep_mean = np.tile(self.dim_means, (self.num_vectors_per_block, 1))     # (50,128)

        # Loop number of blocks
        for i in range(self.num_blocks):

            # Read block -> gives (num_dimensions, num_vectors_per_block) matrix
            X = self._read_block(i)

            # Substract the repmat from block, put into new variable
            Y = np.subtract(X, rep_mean)

            # Update cov matrix: C = C + Y*Y'
            self.cov_matrix = self.cov_matrix + np.matmul(Y.T, Y)
        
        self._close_file()

        # Divide cov matrix by num_vectors
        self.cov_matrix = np.float32(self.cov_matrix / self.num_vectors)

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _calc_transform_matrix(self):

        # KLT transformation matrix is (num_dimensions, num_dimensions)

        # Calculate eigenvalues (array D) and corresponding eigenvectors (matrix V, one eigenvector per column)
        # D is already equivalent to E in MATLAB code.
        D, V = LA.eig(self.cov_matrix)

        # Sort eigenvalues, while keeping original ordering. 
        I = np.argsort(D)

        # Looping over dimensions
        for i in range(self.num_dimensions):

            # Extract eigenvector (looping backwards through original eigenvector ordering). 
            # Tranpose to make it a row vector
            eig_vec = V[:, I[(self.num_dimensions - 1) - i]].T
            
            # Place eigenvector on appropriate row of transform matrix
            self.transform_matrix[i, :] = eig_vec

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _save_dataset_vars(self):
        np.savez(os.path.join(self.path, '') + self.fname + '.dsvars', DIM_MEANS=self.dim_means, COV_MATRIX=self.cov_matrix, TRANSFORM_MATRIX=self.transform_matrix)

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _load_dataset_vars(self):

        print("Loading dataset variables from ", self.path)
        with np.load(os.path.join(self.path, '') + self.fname + '.dsvars.npz') as data:
            self.dim_means = data['DIM_MEANS']
            self.cov_matrix = data['COV_MATRIX']
            self.transform_matrix = data['TRANSFORM_MATRIX']

    #----------------------------------------------------------------------------------------------------------------------------------------
    def process(self):
        
        # Initialisations
        self._initialise()

        if self.dsmode == 'B':

            # Calc dim means
            self._calc_dim_means()

            # Calc cov
            self._calc_cov_matrix()

            # Calc transform
            self._calc_transform_matrix()

            # Save dim_means/cov_matrix/transform_matrix to a file
            self._save_dataset_vars()

    #----------------------------------------------------------------------------------------------------------------------------------------

