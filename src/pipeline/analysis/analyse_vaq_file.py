import matplotlib.pyplot as plt

from pipeline import PipelineElement, TransformationSummary
import numpy as np


class AnalyseVaqFile(PipelineElement):
    def __plot_unique_with_added_dimension(self):
        ax = plt.subplot()
        with self.session.state["VAQ_INDEX_FILE"].open(mode='rb') as f:
            res = f.unsafe_read_all()
            count_unique = [np.unique(res[:i, :], axis=1).shape[1] / res.shape[1] for i in range(1, res.shape[0])]
            ax.plot(list(range(res.shape[0] - 1)), count_unique)
        plt.show()

    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        self.__plot_unique_with_added_dimension()
        return {}
