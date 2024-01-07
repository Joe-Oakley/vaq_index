import os.path

from pipeline import PipelineElement, TransformationSummary
import numpy as np
from pathlib import Path
from qsession import QSession
from vector_file import VectorFile


class KLT(PipelineElement):

    def __calc_dim_means(self):
        running_mean = None
        original_file = self.session.state["ORIGINAL_FILE"]
        with original_file.open(mode="rb") as f:
            for block in f:
                dim_sums = np.divide(np.sum(block, axis=0).reshape(1, original_file.shape[1]),
                                     block.shape[0])
                running_mean = np.add(running_mean, dim_sums) if running_mean is not None else dim_sums
        self.session.state["DIM_MEANS"] = np.divide(running_mean,
                                                    original_file.num_blocks).astype(np.float32)

    def __calc_cov_matrix(self):
        running_cov = None
        original_file = self.session.state["ORIGINAL_FILE"]
        dim_means = self.session.state["DIM_MEANS"]
        with original_file.open(mode="rb") as f:
            for block in f:
                Y = np.subtract(block, dim_means)
                tmp = np.divide(np.matmul(Y.T, Y), block.shape[0])
                running_cov = running_cov + tmp if running_cov is not None else tmp
        self.session.state["COV_MATRIX"] = np.divide(running_cov,
                                                     original_file.num_blocks).astype(np.float32)

    def __calc_transform_matrix(self):
        original_file = self.session.state['ORIGINAL_FILE']
        transform_matrix = np.empty(
            (original_file.shape[1], original_file.shape[1]),
            dtype=np.float32)
        D, V = np.linalg.eig(self.session.state["COV_MATRIX"])
        I = np.argsort(D)
        for i in range(original_file.shape[1]):
            eig_vec = V[:, I[(transform_matrix.shape[0] - 1) - i]].T
            transform_matrix[i, :] = eig_vec
        self.session.state["TRANSFORM_MATRIX"] = transform_matrix

    def __build_transformed_file(self):
        original_file: VectorFile = self.session.state["ORIGINAL_FILE"]
        self.session.state["TRANSFORMED_FILE"] = VectorFile(
            Path(os.path.join(self.session.dataset_path, f"klt_{self.session.fname}")), original_file.shape,
            original_file.dtype, original_file.stored_dtype, original_file.num_blocks)
        with self.session.state["TRANSFORMED_FILE"].open(mode="wb") as tf:
            with self.session.state["ORIGINAL_FILE"].open("rb") as f:
                for block in f:
                    Y = np.subtract(block, self.session.state["DIM_MEANS"])
                    Z = np.matmul(Y, self.session.state["TRANSFORM_MATRIX"])
                    tf.write(Z)

    def __build_transposed_transformed_file(self):
        transformed_file: VectorFile = self.session.state["TRANSFORMED_FILE"]
        self.session.state["TRANSFORMED_TP_FILE"] = VectorFile(
            Path(os.path.join(self.session.dataset_path, f"klt_tp_{self.session.fname}")),
            (transformed_file.shape[1], transformed_file.shape[0]),
            transformed_file.dtype, transformed_file.stored_dtype, transformed_file.shape[1])
        with self.session.state["TRANSFORMED_TP_FILE"].open(mode="wb") as tf:
            with transformed_file.open(mode='rb') as f:
                for i in range(transformed_file.shape[1]):
                    XX = np.zeros(transformed_file.shape[0], dtype=np.float32)
                    ind = 0
                    for block in f:
                        XX[ind:ind + block.shape[0]] = block[:, i]
                        ind += block.shape[0]
                    tf.write(XX)

    def __transform_function(self, block):
        dim_means = self.session.state["DIM_MEANS"]
        transform_matrix = self.session.state['TRANSFORM_MATRIX']
        rep_mean = np.tile(dim_means, (block.shape[0], 1))  # (num_queries, num_dims)
        Y = np.subtract(block, rep_mean)
        return np.matmul(Y, transform_matrix)

    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        self.__calc_dim_means()
        self.__calc_cov_matrix()
        self.__calc_transform_matrix()
        self.__build_transformed_file()
        self.__build_transposed_transformed_file()
        self.session.state["TRANSFORM_FUNCTION"] = self.__transform_function
        return {"created": ("DIM_MEANS", "COV_MATRIX", "TRANSFORM_MATRIX", "TRANSFORM_FUNCTION", "TRANSFORMED_FILE", "TRANSFORMED_TP_FILE")}
