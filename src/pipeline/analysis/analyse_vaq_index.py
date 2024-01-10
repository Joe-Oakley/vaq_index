import math
import random

import matplotlib.pyplot as plt

from pipeline import PipelineElement, TransformationSummary
import numpy as np


class AnalyseVaqIndex(PipelineElement):
    def __plot_unique_with_added_dimension(self):
        ax = plt.subplot()
        with self.session.state["VAQ_INDEX_FILE"].open(mode='rb') as f:
            res = f.unsafe_read_all()
            count_unique = [np.unique(res[:i, :], axis=1).shape[1] / res.shape[1] for i in range(1, res.shape[0])]
            ax.plot(list(range(res.shape[0] - 1)), count_unique)
        plt.show()

    def __plot_points_in_random_dimensions(self, num_dims=16, query_vector_count=1, linewidth=3, figsize=(60, 40),
                                           bins=200):
        tf_tp_dataset = self.session.state["TRANSFORMED_TP_FILE"]
        boundary_vals = self.session.state['BOUNDARY_VALS_MATRIX']
        queryset_vector = self.session.state["TRANSFORMED_QUERYSET"]
        top_k_results = self.session.state["TOP_K_RESULTS"]
        query_vector_indices = random.sample(range(queryset_vector.shape[0]), query_vector_count)
        top_vertex_for_first_query = top_k_results[query_vector_indices[0], 1]
        queryset_vector = queryset_vector[query_vector_indices, :]
        dims = random.sample(range(tf_tp_dataset.shape[0]), num_dims)
        plt_height = math.ceil(math.sqrt(num_dims))
        fig, axs = plt.subplots(plt_height, plt_height, figsize=figsize)
        with tf_tp_dataset.open(mode='rb') as tf_tp_file:
            for plt_index, dim in enumerate(dims):
                dims_vector = tf_tp_file.read_one(dim)
                querset_dims_vector = queryset_vector[:, dim]
                ax = axs[plt_index // plt_height, plt_index % plt_height]
                ax.hist(dims_vector, bins=bins)
                ax.set_xlabel(f"Dimension {dim}")
                for boundary_val in boundary_vals[:, dim]:
                    if np.isnan(boundary_val):
                        break
                    ax.axvline(x=boundary_val, color='black', linestyle='--', linewidth=min(2, linewidth))
                for i, query_dim_element in enumerate(querset_dims_vector):
                    ax.axvline(x=query_dim_element, color='red' if i == 0 else "orange", linestyle='-',
                               linewidth=linewidth)
                ax.axvline(x=top_vertex_for_first_query[dim], color='green', linestyle='-', linewidth=linewidth)

        boundary_line = plt.Line2D([], [], color='black', linestyle='--', linewidth=min(2, linewidth),
                                   label='Boundary points')
        first_query_point = plt.Line2D([], [], color='red', linestyle='-', linewidth=linewidth,
                                       label="First query point")
        other_query_points = plt.Line2D([], [], color='orange', linestyle='-', linewidth=linewidth,
                                        label="Other query points")
        top_1_result = plt.Line2D([], [], color='green', linestyle='-', linewidth=linewidth, label="Top-1 Vector point")
        plt.legend(handles=[boundary_line, first_query_point, other_query_points, top_1_result], loc='upper right',
                   bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig.show()

    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        self.__plot_unique_with_added_dimension()
        self.__plot_points_in_random_dimensions()
        return {}
