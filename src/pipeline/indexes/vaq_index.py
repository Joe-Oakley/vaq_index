from pipeline import PipelineElement, TransformationSummary
from qsession import QSession
import numpy as np
import math
import os
from pathlib import Path
from vector_file import VectorFile


class VAQIndex(PipelineElement):
    MAX_UINT8 = 255
    MAX_LLOYD_ITERATIONS = 75
    LLOYD_STOP = 0.005
    WEIGHTING_FACTOR = 0.1

    def __init__(self, session: QSession, non_uniform_bit_alloc: bool, bit_budget: int, design_boundaries: bool,
                 use_qhist: bool):
        super(VAQIndex, self).__init__(session)
        self.non_uniform_bit_alloc = non_uniform_bit_alloc
        self.bit_budget = bit_budget
        self.design_boundaries = design_boundaries
        self.use_qhist = use_qhist

    def __calculate_energies(self):
        transformed_file = self.session.state["TRANSFORMED_FILE"]
        energies = np.zeros(transformed_file.shape[1], dtype=np.float32)
        with transformed_file.open(mode="r") as f:
            for block in f:
                block = np.square(block)
                energies += np.sum(block, axis=0)
        self.session.state["DIM_ENERGIES"] = energies

    def __allocate_bits(self):
        num_vectors, num_dimensions = self.session.state["ORIGINAL_FILE"].shape
        cells = np.ones(num_dimensions, dtype=np.uint8)
        energies = self.session.state["DIM_ENERGIES"]
        if self.non_uniform_bit_alloc:
            temp_bb = self.bit_budget
            while temp_bb > 0:
                # Get index of dimension with maximum energy
                max_energy_dim = np.min(
                    np.argmax(energies))  # np.min to cater for two dims with equal energy - unlikely!
                # Double the number of "cells" for that dimension
                if cells[max_energy_dim] * 2 > VAQIndex.MAX_UINT8:
                    pass  # Don't blow the capacity of a UINT8
                else:
                    cells[max_energy_dim] = cells[max_energy_dim] * 2
                # Divide the energy of that dimension by 4 - assumes normal distribution.
                energies[max_energy_dim] = energies[max_energy_dim] / 4
                # Check there aren't more cells than data points (unlikely)
                if cells[max_energy_dim] > num_vectors:
                    print("WARNING : self.ctx.cells[max_energy_dim] > self.ctx.num_vectors !!")
                    cells[max_energy_dim] = cells[max_energy_dim] / 2
                else:
                    temp_bb -= 1
        # Uniform bit allocation
        else:
            bits_per_dim = int(self.bit_budget / num_dimensions)
            levels = 2 ** bits_per_dim
            cells *= levels
        self.session.state["DIM_CELLS"] = cells

    def __init_boundaries(self):
        num_vectors, num_dimensions = self.session.state["ORIGINAL_FILE"].shape
        cells = self.session.state["DIM_CELLS"]
        boundary_vals = np.ones((np.max(cells) + 1, num_dimensions), dtype=np.float32) * np.nan
        transformed_tp_file = self.session.state["TRANSFORMED_TP_FILE"]
        with transformed_tp_file.open(mode='r') as f:
            for block_count, block in enumerate(f):
                # Sort the block. N.B., can change np sort algorithm (
                # https://numpy.org/doc/stable/reference/generated/numpy.sort.html)
                sorted_block = np.sort(block, axis=1)
                # Set first boundary_val (0) along current dimension to just less than min value
                boundary_vals[0, block_count] = sorted_block[0, 0] - 0.001
                cells_for_dim = cells[block_count]
                # Loop over the number of cells allocated to current dimension - careful with indices, should start at 1 and go to penultimate.
                # If cells_for_dim = 32, this will go to idx 31. That's fine, because boundary_vals goes up to max(cells) + 1.
                for j in range(1, cells_for_dim):
                    # Using math ceil; alternative is np
                    boundary_vals[j, block_count] = sorted_block[0,
                                                                 math.ceil(j * num_vectors / cells_for_dim)]
                # Set final boundary val along current dim
                # Using idx cells_for_dim is safe since boundary_vals goes up to max(cells) + 1
                boundary_vals[cells_for_dim, block_count] = sorted_block[0, -1] + 0.001
        self.session.state["BOUNDARY_VALS_MATRIX"] = boundary_vals

    def __design_boundaries(self):
        transformed_tp_file = self.session.state["TRANSFORMED_TP_FILE"]
        cells = self.session.state["DIM_CELLS"]
        boundary_vals = self.session.state["BOUNDARY_VALS_MATRIX"]
        with transformed_tp_file.open(mode='r') as f:
            for block_count, block in enumerate(f):
                # Extract cells for current dimension
                cells_for_dim = cells[block_count]

                # If current dimension only has 1 cell (i.e. 0 bits allocated to it), then break and end.
                # Think values in self.cells are implicitly sorted descending.
                if cells[block_count] == 1:
                    break

                # Call Lloyd's algorithm function -> could be replaced by modified Lloyds
                # MATLAB has B(1:CELLS(i)+1, i). Say a dim has 4 cells, this goes from 1 to 5, inclusive.
                # Ours will go from 0 to 5, not inclusive at the top, so really 0,1,2,3,4. Therefore equivalent.
                if self.use_qhist:
                    r, c = self.__weighted_lloyd(block[0], boundary_vals[0:cells_for_dim + 1, block_count])
                else:
                    r, c = self.__lloyd(block[0], boundary_vals[0:cells_for_dim + 1, block_count])

                boundary_vals[0:cells_for_dim + 1, block_count] = c

    def __lloyd(self, block, boundary_vals_in):

        # Inputs:
        # TSET -> block (num_vectors, 1)
        # B(1:CELLS(i)+1,i) -> self.boundary_vals[0:cells_for_dim+1, block_count].
        # First param is the whole unsorted block. Also gets all initialized boundary values for current dim.
        # Dimensions of boundary_vals_in: (cells(i)+1, 1)

        # Init variables
        delta = np.inf
        # stop = 0.001
        c = boundary_vals_in
        M1 = np.min(boundary_vals_in)
        M2 = np.max(boundary_vals_in)
        num_boundary_vals = np.shape(boundary_vals_in)[0]
        r = np.zeros(num_boundary_vals, dtype=np.float32)

        num_lloyd_iterations = 0
        while True:
            delta_new = np.float32(0)
            num_lloyd_iterations += 1

            # Loop over intervals; careful with indices
            for i in range(num_boundary_vals - 1):

                # Find values in block between boundary values; np.where?
                X_i = block[np.where(np.logical_and(block >= c[i], block < c[i + 1]))]

                # If any values found
                if np.shape(X_i)[0] > 0:
                    r[i] = np.mean(X_i)
                else:
                    r[i] = np.random.rand(1) * (M2 - M1) + M1

                # Add representation error over current interval to delta_new
                delta_new += np.sum(np.square(X_i - r[i]))

            # Sort representative values - todo: sorting algorithm selection
            r = np.sort(r)

            # Update boundary values based on representative values
            # for j in range(1, num_boundary_vals):  # MATLAB has a -1 here... don't think we need?   YES WE DO! Otherwise lose top boundary
            for j in range(1, num_boundary_vals - 1):
                c[j] = (r[j - 1] + r[j]) / 2

            # Stopping condition check
            # print("((delta - delta_new)/delta) -> ", ((delta - delta_new)/delta))
            # if ((delta - delta_new) / delta) < stop:
            if (((delta - delta_new) / delta) < VAQIndex.LLOYD_STOP) or (
                    num_lloyd_iterations >= VAQIndex.MAX_LLOYD_ITERATIONS):
                print("Number of Lloyd's iterations: ", str(num_lloyd_iterations))
                return r, c
            delta = delta_new

    def __weighted_lloyd(self, block, boundary_vals_in):

        # Inputs:
        # TSET -> block (num_vectors, 1)
        # B(1:CELLS(i)+1,i) -> self.boundary_vals[0:cells_for_dim+1, block_count].
        # First param is the whole unsorted block. Also gets all initialized boundary values for current dim.
        # Dimensions of boundary_vals_in: (cells(i)+1, 1)

        # Init variables
        delta = np.inf
        # stop = 0.001
        c = boundary_vals_in
        M1 = np.min(boundary_vals_in)
        M2 = np.max(boundary_vals_in)
        num_boundary_vals = np.shape(boundary_vals_in)[0]
        r = np.zeros(num_boundary_vals, dtype=np.float32)

        # print("self.weights: ", self.weights)

        num_lloyd_iterations = 0
        while True:
            delta_new = np.float32(0)
            num_lloyd_iterations += 1

            # Loop over intervals; careful with indices
            for i in range(num_boundary_vals - 1):

                # print("    i loop : ", i)

                # Find values in block between boundary values; np.where?
                idxs = np.where(np.logical_and(block >= c[i], block < c[i + 1]))[0]

                X = block[idxs]
                W = self.weights[idxs]
                W_sum = np.sum(W)
                X_weighted = np.multiply(W, X.ravel())
                X_weighted_sum = np.sum(X_weighted)

                # If any values found
                if np.shape(X)[0] > 0:
                    # r[i] = np.divide(np.sum(np.multiply(W, X)), np.sum(W))
                    r[i] = np.divide(X_weighted_sum, W_sum)
                else:
                    r[i] = np.random.rand(1) * (M2 - M1) + M1

                # Add representation error over current interval to delta_new
                # delta_new += np.sum(np.square(np.multiply(W_i, X_i) - r[i]))
                # delta_new += np.sum(np.square(X_weighted_sum - r[i]))
                delta_new += np.sum(np.square(X_weighted - r[i]))

            # Sort representative values - todo: sorting algorithm selection
            r = np.sort(r)

            # Update boundary values based on representative values
            # for j in range(1, num_boundary_vals):  # MATLAB has a -1 here... don't think we need?   YES WE DO! Otherwise lose top boundary
            for j in range(1, num_boundary_vals - 1):
                c[j] = (r[j - 1] + r[j]) / 2

            # Stopping condition check
            # print("((delta - delta_new)/delta) -> ", ((delta - delta_new)/delta))
            # if ((delta - delta_new) / delta) < stop:
            #     print("Number of Lloyd's iterations: ", str(num_lloyd_iterations))
            #     return r, c
            if (((delta - delta_new) / delta) < VAQIndex.LLOYD_STOP) or (
                    num_lloyd_iterations >= VAQIndex.MAX_LLOYD_ITERATIONS):
                print("Number of Lloyd's iterations: ", str(num_lloyd_iterations))
                return r, c

            delta = delta_new

    def __build_vaq_file(self):
        cells = self.session.state["DIM_CELLS"]
        boundary_vals = self.session.state["BOUNDARY_VALS_MATRIX"]
        transformed_tp_file = self.session.state["TRANSFORMED_TP_FILE"]
        vaq_file = self.session.state["VAQ_INDEX_FILE"] = VectorFile(
            Path(os.path.join(self.session.dataset_path, f"vaq_{self.session.fname}")), transformed_tp_file.shape,
            np.uint8, np.uint8, transformed_tp_file.num_blocks)
        num_dimensions, num_vectors = transformed_tp_file.shape
        with transformed_tp_file.open(mode="r") as f:
            with vaq_file.open(mode='wb') as vaq_f:
                for block_count, block in enumerate(f):
                    cset = np.empty(num_vectors, dtype=np.uint8) * np.nan
                    for i in range(cells[block_count]):
                        l = boundary_vals[i, block_count]
                        r = boundary_vals[i + 1, block_count]
                        A = np.where(np.logical_and(block >= l, block < r))[0]
                        cset[A] = i
                    assert not (cset == np.nan).any(), "Some of the vectors did not find a cell"
                    vaq_f.write(cset)

    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        self.__calculate_energies()
        self.__allocate_bits()
        if self.use_qhist:
            pass
            # self._assign_weights()
        self.__init_boundaries()
        if self.design_boundaries:
            self.__design_boundaries()
        self.__build_vaq_file()
        return {"created": ("DIM_ENERGIES", "DIM_CELLS", "BOUNDARY_VALS_MATRIX", "VAQ_INDEX_FILE")}
