from pipeline import TransformationSummary, PipelineElement
import numpy as np
from qsession import QSession
from scipy.ndimage import minimum_filter1d, maximum_filter1d

class VAQQuerySet(PipelineElement):
    MAX_UINT8 = 255

    def __init__(self, session: QSession, query_k: int):
        super(VAQQuerySet, self).__init__(session)
        self.query_k = query_k

    def __transform_queryset(self):
        dim_means = self.session.state["DIM_MEANS"]
        transform_matrix = self.session.state['TRANSFORM_MATRIX']
        with self.session.state["QUERYSET_FILE"].open(mode="r") as f:
            block = f[0]
            rep_mean = np.tile(dim_means, (block.shape[0], 1))  # (num_queries, num_dims)
            Y = np.subtract(block, rep_mean)
            Z = np.matmul(Y, transform_matrix)
            self.session.state["TRANSFORMED_QUERYSET"] = Z

    def _run_phase_one(self, query_idx):
        Q = self.session.state["TRANSFORMED_QUERYSET"]
        vaq_index_file = self.session.state["VAQ_INDEX_FILE"]
        boundary_vals = self.session.state["BOUNDARY_VALS_MATRIX"]
        cells = self.session.state["DIM_CELLS"]
        num_dimensions, num_vectors = vaq_index_file.shape
        q = Q[query_idx]
        # These are (1, 50) for 50-NN
        UP = np.ones(self.query_k, dtype=np.float32) * np.inf

        L = np.zeros(num_vectors, dtype=np.float32)
        U = np.zeros(num_vectors, dtype=np.float32)
        D = np.square(np.subtract(boundary_vals, q[None, :].ravel()))
        D_MIN = minimum_filter1d(D, size=2, axis=0, origin=-1)
        D_MAX = maximum_filter1d(D, size=2, axis=0, origin=-1)
        with vaq_index_file.open(mode='r') as vaq_f:
            for block_count, block in enumerate(vaq_f):
                cells_for_dim = cells[block_count]
                qj = q[block_count]
                target_cells, = np.where(qj <= boundary_vals[1:cells_for_dim + 1, block_count])
                if target_cells.size == 0:
                    R = cells_for_dim  # If qj > all boundary_vals, put in final cell -> this is the LEFT boundary of final cell.
                else:
                    R = np.min(target_cells)
                L += (D_MIN[block, block_count] * (block != R).astype(np.float32))[0]
                U += D_MAX[block, block_count][0]

        elim = 0

        get_max_next_time = True
        for i in range(num_vectors):
            if get_max_next_time:
                max_up = UP.max()  # https://stackoverflow.com/questions/10943088/numpy-max-or-max-which-one-is-faster
                max_up_idx = np.argmax(UP)  # If multiple same max val, returns min idx.
            if L[i] <= max_up:
                if U[i] <= max_up:
                    UP[max_up_idx] = U[i]
                    get_max_next_time = True
                else:
                    get_max_next_time = False

                    # # Adding elim += 1 makes P1 elims inconsistent with P2 visits.
                    # elim += 1

            else:
                elim += 1

        UP = np.sort(UP, axis=0)
        print(elim)

    def __run_queries(self):
        transformed_queryset = self.session.state["TRANSFORMED_QUERYSET"]
        for i in range(transformed_queryset.shape[0]):
            self._run_phase_one(i)

    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        self.__transform_queryset()
        self.__run_queries()
