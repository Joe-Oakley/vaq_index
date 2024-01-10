from pipeline import TransformationSummary, PipelineElement
import numpy as np
from qsession import QSession
from scipy.ndimage import minimum_filter1d, maximum_filter1d
import matplotlib.pyplot as plt


class VAQQuerySet(PipelineElement):
    MAX_UINT8 = 255

    def __init__(self, session: QSession, query_k: int):
        super(VAQQuerySet, self).__init__(session)
        self.query_k = query_k

    def __transform_queryset(self):
        with self.session.state["QUERYSET_FILE"].open(mode="rb") as f:
            self.session.state["TRANSFORMED_QUERYSET"] = self.session.state["TRANSFORM_FUNCTION"](f[0])

    def _run_phase_one(self, query_idx):
        Q = self.session.state["TRANSFORMED_QUERYSET"]
        vaq_index_file = self.session.state["VAQ_INDEX_FILE"]
        boundary_vals = self.session.state["BOUNDARY_VALS_MATRIX"]
        cells = self.session.state["DIM_CELLS"]
        num_dimensions, num_vectors = vaq_index_file.shape
        q = Q[query_idx]
        # These are (1, 50) for 50-NN
        L = np.zeros(num_vectors, dtype=np.float32)
        U = np.zeros(num_vectors, dtype=np.float32)
        D = np.square(np.subtract(boundary_vals, q[None, :].ravel()))
        D_MIN = minimum_filter1d(D, size=2, axis=0, origin=-1)
        D_MAX = maximum_filter1d(D, size=2, axis=0, origin=-1)
        with vaq_index_file.open(mode='rb') as vaq_f:
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
        return L, U

    def _run_phase_two(self, query_idx, L, U) -> int:
        top_k_results = self.session.state["TOP_K_RESULTS"]
        transformed_dataset = self.session.state['TRANSFORMED_FILE']
        J = np.argsort(L, axis=0)
        LL = np.sort(L, axis=0)
        ANS = np.ones(self.query_k, dtype=np.float32) * np.inf
        V = np.ones(self.query_k, dtype=np.int64)
        num_dimensions, num_vectors = self.session.state["VAQ_INDEX_FILE"].shape
        q = self.session.state["TRANSFORMED_QUERYSET"][query_idx]

        # Loop over all vectors; is this sensible; don't we just want to consider candidates only in terms of their LBs?
        vectors_accessed_p2 = 0
        with transformed_dataset.open(mode="rb") as tr_file:
            for i in range(num_vectors):

                # If lower bound of i is greater than 50th best upper bound, stop
                if LL[i] > ANS[self.query_k - 1]:
                    break
                else:
                    # Random read (of num_dimensions words) from transformed file.
                    read_vector = tr_file.read_one(J[i])

                    # Append squared distance between self.q (could also use self.Q[query_idx, :]) and the vector read from disk to ANS
                    # self.q and TSET are both (1, num_dimensions)
                    T = np.append(ANS, np.sum(np.square(q - read_vector)))

                    # Append J[i] to V
                    W = np.append(V, J[i])

                    # Sort ANS and also keep original locations
                    I = np.argsort(T)
                    T = np.sort(T)

                    # Trim ANS to only first query_k
                    ANS = T[0:self.query_k]

                    # V=W(I(1:QUERY_SIZE)); think these are the indices of answers?
                    V = W[I[0:self.query_k]]

                    # Increment counter; not using i since we'll lose it after the loop
                    vectors_accessed_p2 += 1
            for i, ind in enumerate(V):
                top_k_results[query_idx, i, :] = tr_file.read_one(ind)

        return vectors_accessed_p2

    def __run_queries(self):
        transformed_queryset = self.session.state["TRANSFORMED_QUERYSET"]
        phase_2_vectors_accessed = self.session.state["PHASE_2_VECTORS_ACCESSED"]
        for i in range(transformed_queryset.shape[0]):
            L, U = self._run_phase_one(i)
            phase_2_vectors_accessed[i] = self._run_phase_two(i, L, U)
        print(f"Average number of vectors accessed for phase 2 is %d" % (phase_2_vectors_accessed.mean(),))

    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        self.session.state["PHASE_2_VECTORS_ACCESSED"] = np.zeros(self.session.state["QUERYSET_FILE"].shape[0])
        self.session.state["TOP_K_RESULTS"] = np.zeros((self.session.state["QUERYSET_FILE"].shape[0], self.query_k,
                                                        self.session.state["QUERYSET_FILE"].shape[1]))
        self.__transform_queryset()
        self.__run_queries()
        return {"created": ("TRANSFORMED_QUERYSET", "PHASE_2_VECTORS_ACCESSED", "TOP_K_RESULTS")}
