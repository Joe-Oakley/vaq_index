from base import Transformation
import numpy as np


class KLTTransformation(Transformation):

    def __calc_dim_means(self):
        running_mean = None
        with self.session.state["ORIGINAL_FILE"].open(mode="rb") as f:
            for block in f:
                dim_sums = np.divide(np.sum(block, axis=0).reshape(1, self.session.shape[1]),
                                     block.shape[0])
                running_mean = np.add(running_mean, dim_sums) if running_mean is not None else dim_sums
        self.session.state["DIM_MEANS"] = np.divide(running_mean,
                                                    self.session.state["ORIGINAL_FILE"].num_blocks).astype(np.float32)

    def __calc_cov_matrix(self):
        running_cov = None
        with self.session.state["ORIGINAL_FILE"].open(mode="rb") as f:
            for block in f:
                rep_mean = np.tile(self.session.state["DIM_MEANS"], block.shape[0])
                Y = np.subtract(block, rep_mean)
                tmp = np.divide(np.matmul(Y.T, Y), block.shape[0])
                running_cov = running_cov + tmp if running_cov is not None else tmp
        self.session.state["COV_MATRIX"] = np.divide(running_cov,
                                                     self.session.state["ORIGINAL_FILE"].num_blocks).astype(np.float32)

    def _calc_transform_matrix(self):
        transform_matrix = np.empty((self.session.shape[1], self.session.shape[1]), dtype=np.float32)
        D, V = np.linalg.eig(self.session.state["COV_MATRIX"])
        I = np.argsort(D)
        for i in range(self.session.shape[1]):
            eig_vec = V[:, I[(self.session.shape[1] - 1) - i]].T
            transform_matrix[i, :] = eig_vec
        self.session.state["TRANSFORM_MATRIX"] = transform_matrix

    def process(self, pipe_state=None):
        pass
