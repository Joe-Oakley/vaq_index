import numpy as np
from numpy import linalg as LA
import os

from qsession import QSession

class DataSet:
    def __init__(self, ctx: QSession = None):
        self.ctx            = ctx
        self.full_fname     = None
        self.file_handle    = None

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _open_file(self):
        if self.file_handle is not None and not self.file_handle.closed:
            self.file_handle.close()
        self.file_handle = open(self.full_fname, mode="rb")

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self):
        self.file_handle.close()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialise(self):
        np.set_printoptions(suppress=True)
        self.full_fname = os.path.join(self.ctx.path, '') + self.ctx.fname

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _read_block(self, block_idx):

        #(num_dims+1, block_size/(num_dims+1)) -> (num_dims, num_vectors_per_block) if all as float32

        self.file_handle.seek(self.ctx.num_words_per_block*block_idx*self.ctx.word_size, os.SEEK_SET) # Multiply by word_size since seek wants a byte location.
        if self.ctx.big_endian:
            block = np.fromfile(file=self.file_handle, count=self.ctx.num_words_per_block, dtype=np.float32).byteswap(inplace=True)
        else:
            block = np.fromfile(file=self.file_handle, count=self.ctx.num_words_per_block, dtype=np.float32)

        block = np.reshape(block, (self.ctx.num_vectors_per_block, self.ctx.num_dimensions+1), order="C")
        block = np.delete(block, 0, 1)
        
        return block
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    def generate_dataset_block(self,start_offset=0):

        block_idx = start_offset
        with open(self.full_fname, mode="rb") as f:
            
            while True:
                f.seek(self.ctx.num_words_per_block*block_idx*self.ctx.word_size, os.SEEK_SET) # Multiply by word_size since seek wants a byte location.
                if self.ctx.big_endian:
                    block = np.fromfile(file=f, count=self.ctx.num_words_per_block, dtype=np.float32).byteswap(inplace=True)
                else:
                    block = np.fromfile(file=f, count=self.ctx.num_words_per_block, dtype=np.float32)

                if block.size > 0:
                    block = np.reshape(block, (self.ctx.num_vectors_per_block, self.ctx.num_dimensions+1), order="C")
                    block = np.delete(block, 0, 1)
                    # AT THIS POINT, EACH COL IS A VECTOR AND EACH ROW IS AN ATTRIBUTE (FOR 1/NUM_BLOCKS OF THE VECTORS)

                    yield block
                    block_idx +=1
                else:
                    break

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _calc_dim_means(self):

        self._open_file()
        # Loop over number of blocks
        for i in range(self.ctx.num_blocks):
            block = self._read_block(i)
            dim_sums = np.divide(np.sum(block, axis=0).reshape(1, self.ctx.num_dimensions), self.ctx.num_vectors_per_block)
            self.ctx.dim_means = np.add(self.ctx.dim_means, dim_sums)

        self._close_file()
        # self.ctx.dim_means = np.divide(self.ctx.dim_means, self.ctx.num_blocks) # No longer casting to np.float32
        self.ctx.dim_means = np.divide(self.ctx.dim_means, self.ctx.num_blocks).astype(np.float32) # Oh yes it did
        

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # self.cov_matrix is (num_dimensions, num_dimensions)
    def _calc_cov_matrix(self):
        
        # Open file
        self._open_file()

        # Generate repmat(dim_means, 1, num_vectors_per_block)
        rep_mean = np.tile(self.ctx.dim_means, (self.ctx.num_vectors_per_block, 1))  # (50,128)

        # Loop number of blocks
        for i in range(self.ctx.num_blocks):
            # Read block -> gives (num_dimensions, num_vectors_per_block) matrix
            X = self._read_block(i)

            # Substract the repmat from block, put into new variable
            Y = np.subtract(X, rep_mean)

            # Update cov matrix: C = C + Y*Y'
            self.ctx.cov_matrix = self.ctx.cov_matrix + np.divide(np.matmul(Y.T, Y), self.ctx.num_vectors_per_block)

        self._close_file()

        self.ctx.cov_matrix = np.divide(self.ctx.cov_matrix, self.ctx.num_blocks).astype(np.float32)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Transformation matrix is (num_dimensions, num_dimensions)
    def _calc_transform_matrix(self):

        # Calculate eigenvalues (array D) and corresponding eigenvectors (matrix V, one eigenvector per column)
        # D is already equivalent to E in MATLAB code.
        D, V = LA.eig(self.ctx.cov_matrix)

        # Sort eigenvalues, while keeping original ordering. 
        I = np.argsort(D)

        # Looping over dimensions
        for i in range(self.ctx.num_dimensions):
            # Extract eigenvector (looping backwards through original eigenvector ordering).
            # Tranpose to make it a row vector
            eig_vec = V[:, I[(self.ctx.num_dimensions - 1) - i]].T

            # Place eigenvector on appropriate row of transform matrix
            self.ctx.transform_matrix[i, :] = eig_vec

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _save_dataset_vars(self):

        np.savez(os.path.join(self.ctx.path, '') + self.ctx.fname + '.dsvars', DIM_MEANS=self.ctx.dim_means,
                 COV_MATRIX=self.ctx.cov_matrix, TRANSFORM_MATRIX=self.ctx.transform_matrix)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_dataset_vars(self):

        print("Loading dataset variables from ", self.ctx.path)
        with np.load(os.path.join(self.ctx.path, '') + self.ctx.fname + '.dsvars.npz') as data:
            self.ctx.dim_means = data['DIM_MEANS']
            self.ctx.cov_matrix = data['COV_MATRIX']
            self.ctx.transform_matrix = data['TRANSFORM_MATRIX']

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def process(self):

        # Initialisations
        self._initialise()

        # Calc dim means
        self._calc_dim_means()

        # Calc cov
        self._calc_cov_matrix()

        # Calc transform
        self._calc_transform_matrix()

        # Save dim_means/cov_matrix/transform_matrix to a file
        self._save_dataset_vars()

    # ----------------------------------------------------------------------------------------------------------------------------------------
