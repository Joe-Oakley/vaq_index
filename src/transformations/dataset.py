import numpy as np
from numpy import linalg as LA
import os
from pathlib import Path
from qsession import QSession
from vector_file import VectorFile

class DataSet:
    def __init__(self, ctx: QSession = None):
        self.ctx = ctx
        self.file = VectorFile(Path(os.path.join(self.ctx.dataset_path, '') + self.ctx.fname), self.ctx.shape, big_endian=self.ctx.big_endian, offsets=(1, 0))
        self.full_fname = None

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _calc_dim_means(self):
        with self.file.open(mode="rb") as f:
            for block in f:
                dim_sums = np.divide(np.sum(block, axis=0).reshape(1, self.ctx.num_dimensions),
                                     block.shape[0])
                self.ctx.dim_means = np.add(self.ctx.dim_means, dim_sums)
        self.ctx.dim_means = np.divide(self.ctx.dim_means, self.ctx.num_blocks).astype(np.float32)  # Oh yes it did

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # self.cov_matrix is (num_dimensions, num_dimensions)
    def _calc_cov_matrix(self):

        # Open file
        # self._open_file()
        with open(self.full_fname, mode='rb') as f:
            # Generate repmat(dim_means, 1, num_vectors_per_block)
            rep_mean = np.tile(self.ctx.dim_means, (self.ctx.num_vectors_per_block, 1))  # (50,128)

            # Loop number of blocks
            for i in range(self.ctx.num_blocks):
                # Read block -> gives (num_dimensions, num_vectors_per_block) matrix
                X = self._read_block(i, f)

                # Substract the repmat from block, put into new variable
                Y = np.subtract(X, rep_mean)

                # Update cov matrix: C = C + Y*Y'
                self.ctx.cov_matrix = self.ctx.cov_matrix + np.divide(np.matmul(Y.T, Y), self.ctx.num_vectors_per_block)

        # self._close_file()

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

        np.savez(os.path.join(self.ctx.dataset_path, '') + self.ctx.fname + '.dsvars', DIM_MEANS=self.ctx.dim_means,
                 COV_MATRIX=self.ctx.cov_matrix, TRANSFORM_MATRIX=self.ctx.transform_matrix)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_dataset_vars(self):

        print("Loading dataset variables from ", self.ctx.dataset_path)
        with np.load(os.path.join(self.ctx.dataset_path, '') + self.ctx.fname + '.dsvars.npz') as data:
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
