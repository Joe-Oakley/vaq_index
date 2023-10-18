import numpy as np
from numpy import linalg as LA
import os
import json
import math

from transformed import TransformedDataSet


class VAQIndex:
    MAX_UINT8 = 255

    def __init__(self, q_lambda=1, vaqmode='B', bit_budget=0, non_uniform_bit_alloc=True, design_boundaries=True,
                 use_query_hist=True, tf_dataset: TransformedDataSet = None):
        self.full_vaq_fname = None
        self.q_lambda = q_lambda
        self.vaqmode = vaqmode
        self.bit_budget = bit_budget
        self.TDS = tf_dataset
        self.non_uniform_bit_alloc = non_uniform_bit_alloc
        self.design_boundaries = design_boundaries
        self.use_query_hist = use_query_hist

        self.vaq_handle_read = None
        self.vaq_handle_write = None

        self.energies = None
        self.cells = None
        self.boundary_vals = None
        self.cset = None

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _open_file(self, mode):

        if mode == 'rb':
            self.vaq_handle_read = open(self.full_vaq_fname, mode=mode)
        elif mode == 'wb':
            self.vaq_handle_write = open(self.full_vaq_fname, mode=mode)
        else:
            raise ValueError("Invalid mode selected: ", mode)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self, handle):

        if handle == self.vaq_handle_read:
            self.vaq_handle_read = None
        elif handle == self.vaq_handle_write:
            self.vaq_handle_write = None
        else:
            raise ValueError("Invalid handle given to _close_file().")

        handle.close()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialise(self):

        # Todo: VAQ modes!
        print("***************************************************")
        print("***************************************************")
        print("ARRIVED AT VAQINDEX INIT")
        print("***************************************************")
        print("***************************************************")

        # Build full filename
        self.full_vaq_fname = os.path.join(self.TDS.DS.path, '') + self.TDS.DS.fname + '.vaq'

        # Setup vars for energy and cells
        self.energies = np.zeros(self.TDS.DS.num_dimensions, dtype=np.float32)
        self.cells = np.ones(self.TDS.DS.num_dimensions, dtype=np.uint8)
        self.cset = np.ones(self.TDS.DS.num_vectors, dtype=np.float32)
        self.cset = np.ones(self.TDS.DS.num_vectors, dtype=np.uint8)

        if self.non_uniform_bit_alloc == False:
            assert self.bit_budget % self.TDS.DS.num_dimensions == 0, "Bit budget cannot be evenly divided among dimensions (uniform bit allocation)."

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Uses transformed file 
    def _calc_energies(self):

        # Init tf generator
        tf_gene = self.TDS.generate_tf_block()

        # Tf generator block loop - each block should be (num_dims, num_vectors_per_block)
        for block in tf_gene:
            # Element wise square of block
            block = np.square(block)

            # Sum along columns -> add to energies. Energies is (1,num_dims)
            self.energies += np.sum(block, axis=0)
        dummy = 0

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _allocate_bits(self):

        if self.non_uniform_bit_alloc:

            temp_bb = self.bit_budget
            while temp_bb > 0:
                # Get index of dimension with maximum energy
                max_energy_dim = np.min(
                    np.argmax(self.energies))  # np.min to cater for two dims with equal energy - unlikely!

                # Double the number of "cells" for that dimension
                if self.cells[max_energy_dim] * 2 > VAQIndex.MAX_UINT8:
                    pass  # Don't blow the capcity of a UINT8
                else:
                    self.cells[max_energy_dim] = self.cells[max_energy_dim] * 2

                # Divide the energy of that dimension by 4 (for future iterations)                
                self.energies[max_energy_dim] = self.energies[max_energy_dim] / 4

                # Check there aren't more cells than data points (unlikely)
                if self.cells[max_energy_dim] > self.TDS.DS.num_vectors:
                    print("WARNING : self.cells[max_energy_dim] > self.TDS.DS.num_vectors !!")
                    self.cells[max_energy_dim] = self.cells[max_energy_dim] / 2

                # Decrement temp_bb
                else:
                    temp_bb -= 1

        # Uniform bit allocation - have already asserted that bit budget divides by num_dims
        else:
            bits_per_dim = int(self.bit_budget / self.TDS.DS.num_dimensions)
            levels = 2 ** bits_per_dim

            self.cells *= levels
        dummy = 0
        # ----------------------------------------------------------------------------------------------------------------------------------------

    # Uses transposed file. Initializes boundary values such that cells are equally populated.
    def _init_boundaries(self):

        self.boundary_vals = np.zeros((np.max(self.cells) + 1, self.TDS.DS.num_dimensions), dtype=np.float32)

        # Set up tp generator
        tp_gene = self.TDS.generate_tp_block()

        # Set up block counter. Also effectively a dimension counter.
        block_count = 0

        # Loop over tp generated blocks
        # Each tp block is (num_vectors, 1) of np.float32. One block = all values for 1 dimension.
        for block in tp_gene:

            # Sort the block. N.B., can change np sort algorithm (https://numpy.org/doc/stable/reference/generated/numpy.sort.html)
            sorted_block = np.sort(block, axis=0)

            # Set first boundary_val (0) along current dimension to min(block) - 0.001; just less than min value
            # along dimension N.B. self.boundary_vals is (max(self.cells) + 1, num_dimensions). +1 as there are k+1
            # boundary values for k cells.
            self.boundary_vals[0, block_count] = sorted_block[0] - 0.001

            cells_for_dim = self.cells[block_count]

            # Loop over the number of cells allocated to current dimension - careful with indices, should start at 1 and go to penultimate.
            # If cells_for_dim = 32, this will go to idx 31. That's fine, because boundary_vals goes up to max(cells) + 1.
            for j in range(1, cells_for_dim):
                # Set boundary vals
                # Not adding 1 to j idx, because we start at 0 in Python.
                # Using math ceil; alternative is np
                self.boundary_vals[j, block_count] = sorted_block[
                    math.ceil(j * self.TDS.DS.num_vectors / cells_for_dim)]

            # Set final boundary val along current dim
            # Using idx cells_for_dim is safe since boundary_vals goes up to max(cells) + 1
            self.boundary_vals[cells_for_dim, block_count] = sorted_block[self.TDS.DS.num_vectors - 1] + 0.001

            # Increment block_count
            block_count += 1
        dummy = 0

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _design_boundaries(self):
        # Operates on transposed file

        # Set up tp generator
        tp_gene = self.TDS.generate_tp_block()

        # Set up block counter, this is also dimension counter
        block_count = 0

        # Loop over blocks (i.e. dimensions). Each block is (num_vectors, 1)
        for block in tp_gene:

            # Extract cells for current dimension
            cells_for_dim = self.cells[block_count]

            # If current dimension only has 1 cell (i.e. 0 bits allocated to it), then break and end.
            # We should break rather than continue, because I'm pretty sure values in self.cells are implicitly sorted descending?
            if self.cells[block_count] == 1:
                break

            # Call Lloyd's algorithm function -> could be replaced by modified Lloyds
            # MATLAB has B(1:CELLS(i)+1, i). Say a dim has 32 cells, this goes from 1 to 33, inclusive.
            # Ours will go from 0 to 33, not inclusive at the top, so really 0->32. Should be equivalent.

            # Experiment 1
            # r, c = self._lloyd(block, self.boundary_vals[0:cells_for_dim+1, block_count])   
            r, c = self._lloyd(block, self.boundary_vals[0:cells_for_dim, block_count])

            # Set boundary values to designed boundary values. Not using r; might stop returning it.

            # Experiment 1
            # self.boundary_vals[0:cells_for_dim+1, block_count] = c
            self.boundary_vals[0:cells_for_dim, block_count] = c

            # Increment block_count - this was missing
            block_count += 1
        dummy = 0

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _lloyd(self, block, boundary_vals_in):

        # Inputs: 
        # TSET -> block (num_vectors, 1)
        # B(1:CELLS(i)+1,i) -> self.boundary_vals[0:cells_for_dim+1, block_count].
        # First param is the whole unsorted block. Also gets all initialized boundary values for current dim.
        # Dimensions of boundary_vals_in: (cells(i)+1, 1)

        # Init variables
        delta = np.inf
        stop = 0.001
        c = boundary_vals_in
        M1 = np.min(boundary_vals_in)
        M2 = np.max(boundary_vals_in)
        num_boundary_vals = np.shape(boundary_vals_in)[0]
        r = np.zeros(num_boundary_vals, dtype=np.float32)

        num_lloyd_iterations = 0
        while True:
            delta_new = np.float32(0)
            num_lloyd_iterations += 1

            # print("_lloyd while true loop")
            # print()

            # Loop over intervals; careful with indices
            for i in range(num_boundary_vals - 1):

                # print("    i loop : ", i)                

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

            # print()

            # Update boundary values based on representative values
            for j in range(1, num_boundary_vals):  # MATLAB has a -1 here... don't think we need?

                # print("    j loop : ", j)                

                c[j] = (r[j - 1] + r[j]) / 2

            # Stopping condition check
            # print("((delta - delta_new)/delta) -> ", ((delta - delta_new)/delta))
            if ((delta - delta_new) / delta) < stop:
                # print("Number of Lloyd's iterations: ", str(num_lloyd_iterations))
                return r, c
            delta = delta_new

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Operates on the transposed datafile. 
    # Writes the vaqfile -> use vaq_handle_write     
    def _create_vaqfile(self):

        # Setup tp generator
        tp_gene = self.TDS.generate_tp_block()

        # Setup tp block counter -> also dimension counter
        block_count = 0

        # Open vaqfile -> write handle
        self._open_file('wb')

        # Loop over tp blocks (i.e. loop over dimensions)
        for block in tp_gene:

            print("_create_vaqfile -> Dimension : ", block_count)

            # Loop over i in range(cells[block_count])
            for i in range(self.cells[block_count]):
                print("                   Block     : ", i)

                # A = np where statement, to find values between boundary_vals i and i+1 along current dim (idx block_count)
                # Effectively finding the indices of all elements that lie between the two boundaries, i.e. in a particular cell.
                l = self.boundary_vals[i, block_count]
                r = self.boundary_vals[i + 1, block_count]
                A = np.where(np.logical_and(block >= l, block < r))[0]

                # MATLAB: Set CSET of those indices to the k-1. Effectively, if a record lies between the 1st and 2nd boundary value, assign it 
                # to the 0th cells (as this is really the cell bounded by boundary values 1 and 2.)
                # Python: Set it to k, rather than k-1. If it lies between boundary values 0 and 1, put it in cell 0.
                self.cset[A] = i

            # Write CSET to vaq_handle_write
            self.vaq_handle_write.write(self.cset)

            block_count += 1

        # Close vaqfile
        self._close_file(self.vaq_handle_write)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Gives (num_vectors, 1) block of vaq_index; all data for a single dimension.    
    def generate_vaq_block(self, start_offset=0):

        block_idx = start_offset

        # Reading a column of VAQ index per block. Each word (usually 4 bytes) contains 4 VAQ cells
        with open(self.full_vaq_fname, mode="rb") as f:

            while True:
                # Using TDS.tp_num_words_per_block as should be same size
                f.seek(self.TDS.tp_num_words_per_block * block_idx, os.SEEK_SET)
                block = np.fromfile(file=f, count=self.TDS.tp_num_words_per_block, dtype=np.uint8)

                if block.size > 0:
                    block = np.reshape(block, (self.TDS.DS.num_vectors, 1))
                    yield block
                    block_idx += 1
                else:
                    break

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _save_vaq_vars(self):
        pass

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_vaqfile(self):
        pass

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_vaq_vars(self):
        pass

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def build(self):
        # Initialisations
        self._initialise()

        # Calculate energies
        self._calc_energies()

        # Bit allocation (covers uniform + non-uniform)
        self._allocate_bits()

        # Init boundary values
        self._init_boundaries()

        # Design boundary values (with Lloyd's)
        self._design_boundaries()

        # Create vaqfile
        self._create_vaqfile()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def load(self):
        pass

    # #----------------------------------------------------------------------------------------------------------------------------------------
    # def vaq_generate_column(self):
    #     pass

    # #----------------------------------------------------------------------------------------------------------------------------------------
    # def vaq_generate_column_memory(self):
    #     pass
