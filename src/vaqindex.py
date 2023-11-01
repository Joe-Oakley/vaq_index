import numpy as np
from numpy import linalg as LA
import os
import json
import math

from qsession import QSession

class VAQIndex:
    MAX_UINT8 = 255

    def __init__(self, ctx: QSession = None):
        self.ctx                  = ctx
        self.full_vaq_fname       = None 
        # self.full_qhist_fname     = None # Should this be in qsession?
        self.full_weights_fname   = None # Writing weights will be useful when updating VAQ later.
        self.vaq_handle_read      = None
        self.vaq_handle_write     = None
        self.qhist_handle_read    = None
        self.qhist_handle_write   = None
        self.weights_handle_read  = None
        self.weights_handle_write = None
        self.energies             = None
        self.cset                 = None
        self.vaqdata              = None
        self.weights              = None # Probably only needed here
        
        # Initialisations
        self._initialise()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _open_file(self, ftype, mode):

        if ftype == 'vaq':
            if mode == 'rb':
                self.vaq_handle_read = open(self.full_vaq_fname, mode=mode)
            elif mode == 'wb':
                self.vaq_handle_write = open(self.full_vaq_fname, mode=mode)
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'qhist':
            if mode == 'rb':
                self.qhist_handle_read = open(self.ctx.qhist_fname, mode=mode)
            elif mode == 'wb':
                self.qhist_handle_write = open(self.ctx.qhist_fname, mode=mode)
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'weights':
            if mode == 'rb':
                self.weights_handle_read = open(self.full_weights_fname, mode=mode)
            elif mode == 'wb':
                self.weights_handle_write = open(self.full_weights_fname, mode=mode)
            else:
                raise ValueError("Invalid mode selected: ", mode)
        else:
            raise ValueError("Invalid ftype selected: ", ftype)

    # # ----------------------------------------------------------------------------------------------------------------------------------------
    # def _close_file(self, handle):

    #     if handle == self.vaq_handle_read:
    #         self.vaq_handle_read = None
    #     elif handle == self.vaq_handle_write:
    #         self.vaq_handle_write = None
    #     elif handle == self.qhist_handle_read:
    #         self.qhist_handle_read = None
    #     elif handle == self.qhist_handle_write:
    #         self.qhist_handle_write = None
    #     elif handle == self.weights_handle_read:
    #         self.weights_handle_read = None
    #     elif handle == self.weights_handle_write:
    #         self.weights_handle_write = None
    #     else:
    #         raise ValueError("Invalid handle given to _close_file().")

    #     handle.close()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self, ftype, mode): # ftypes: vaq, qhist, weights. modes: 'rb', 'wb'

        if ftype == 'vaq':
            if mode == 'rb':
                self.vaq_handle_read.close()
                self.vaq_handle_read = None
            elif mode == 'wb':
                self.vaq_handle_write.close()
                self.vaq_handle_write = None
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'qhist':
            if mode == 'rb':
                self.qhist_handle_read.close()
                self.qhist_handle_read = None
            elif mode == 'wb':
                self.qhist_handle_write.close()
                self.qhist_handle_write = None
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'weights':
            if mode == 'rb':
                self.weights_handle_read.close()
                self.weights_handle_read = None
                
            if mode == 'wb':
                self.weights_handle_write.close()
                self.weights_handle_write = None
            else:
                raise ValueError("Invalid mode selected: ", mode)
        else:
            raise ValueError("Invalid ftype selected: ", ftype)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialise(self):

        # Set full filename
        self.full_vaq_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.vaq'
        
        if self.ctx.non_uniform_bit_alloc == False:
            assert self.ctx.bit_budget % self.ctx.num_dimensions == 0, "Bit budget cannot be evenly divided among dimensions (uniform bit allocation)."
            
        # For query-only runs, load CELLS and BOUNDARY_VALS from file saved during VAQ build
        if self.ctx.mode == 'Q':
            self._load_vaq_vars()

            # If in-memory vaq file requested for mode Q, load it now.
            # N.B., for mode F it will be loaded after build phase (later)
            if self.ctx.inmem_vaqdata:
                self._load_vaqdata()

        if self.ctx.use_qhist:
            self.full_weights_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.weights'

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Uses transformed file 
    def _calc_energies(self):

        self.energies = np.zeros(self.ctx.num_dimensions, dtype=np.float32)

        # Init tf generator
        tf_gene = self.ctx.TDS.generate_tf_block()

        # Tf generator block loop - each block should be (num_dims, num_vectors_per_block)
        for block in tf_gene:
            # Element wise square of block
            block = np.square(block)

            # Sum along columns -> add to energies. Energies is (1,num_dims)
            self.energies += np.sum(block, axis=0)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _allocate_bits(self):

        self.ctx.cells = np.ones(self.ctx.num_dimensions, dtype=np.uint8)  

        if self.ctx.non_uniform_bit_alloc:

            temp_bb = self.ctx.bit_budget
            while temp_bb > 0:
                # Get index of dimension with maximum energy
                max_energy_dim = np.min(np.argmax(self.energies))  # np.min to cater for two dims with equal energy - unlikely!

                # Double the number of "cells" for that dimension
                if self.ctx.cells[max_energy_dim] * 2 > VAQIndex.MAX_UINT8:
                    pass  # Don't blow the capacity of a UINT8
                else:
                    self.ctx.cells[max_energy_dim] = self.ctx.cells[max_energy_dim] * 2

                # Divide the energy of that dimension by 4 - assumes normal distribution.              
                self.energies[max_energy_dim] = self.energies[max_energy_dim] / 4

                # Check there aren't more cells than data points (unlikely)
                if self.ctx.cells[max_energy_dim] > self.ctx.num_vectors:
                    print("WARNING : self.ctx.cells[max_energy_dim] > self.ctx.num_vectors !!")
                    self.ctx.cells[max_energy_dim] = self.ctx.cells[max_energy_dim] / 2

                else:
                    temp_bb -= 1

        # Uniform bit allocation
        else:
            bits_per_dim = int(self.ctx.bit_budget / self.ctx.num_dimensions)
            levels = 2 ** bits_per_dim

            self.ctx.cells *= levels

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _assign_weights(self):

        # Instantiate weights array: np ones, (num_vectors, 1)
        self.weights = np.ones(self.ctx.num_vectors, dtype=np.float32)

        print("weights before: " , self.weights)

        # Open qhist file (read)
        self._open_file('qhist', 'rb')

        # Read first word -> num_queries (in qhist file)
        byte_counter = 0
        self.qhist_handle_read.seek(byte_counter, os.SEEK_SET) # Seek takes a byte counter
        num_queries_qhist = np.fromfile(file=self.qhist_handle_read, count=1, dtype=np.uint32)[0] # Count is in words
        byte_counter += self.ctx.word_size

        for i in range(num_queries_qhist):

            # Read 4 words (np.fromfile): query_id, query_k, phase 1 elims, phase 2 visits
            self.qhist_handle_read.seek(byte_counter, os.SEEK_SET)
            q_info = np.fromfile(file=self.qhist_handle_read, count = 4, dtype=np.uint32)
            q_info = np.reshape(q_info, (4,1), order="C")
            q_id, q_k, q_p1, q_p2 = q_info[0,0], q_info[1,0], q_info[2,0], q_info[3,0] 
            byte_counter += 4 * self.ctx.word_size

            # Read next (2*query_k) words (np.fromfile) -> reshape into (k, 2) distances matrix, probably order C
            self.qhist_handle_read.seek(byte_counter, os.SEEK_SET)
            distances = np.fromfile(file=self.qhist_handle_read, count = 2*q_k, dtype=np.float32)
            distances = np.reshape(distances, (q_k, 2), order='C')
            byte_counter += 2 * q_k * self.ctx.word_size

            # Loop over rows k of distances matrix
            for j in range(q_k):

                # Calculate query visit ratio for this query
                qvr = np.divide(q_p2, q_k)

                # Update self.weights[k]
                vector_id = int(distances[j,0])

                if self.ctx.relative_dist:
                    if distances[0,1] > 0: 
                        self.weights[vector_id] += (np.divide(distances[0,1], distances[j,1]) * self.ctx.q_lambda * qvr)
                    else:
                        self.weights[vector_id] += (np.divide(0.001, distances[j,1]) * self.ctx.q_lambda * qvr)
                else:
                    self.weights[vector_id] += self.ctx.q_lambda * qvr

        # print("weights after: " , self.weights)
        # print("mean of weights after: ", np.mean(self.weights))
        # print("max of weights after: ", np.max(self.weights))
        
        # Close qhist fle
        self._close_file('qhist', 'rb')

        # Open weights file (write)
        self._open_file('weights', 'wb')

        # Write weights
        self.weights_handle_write.write(self.weights)

        # Close weights file
        self._close_file('weights', 'wb')


    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Uses transposed file. Initializes boundary values such that cells are equally populated.
    def _init_boundaries(self):

        self.ctx.boundary_vals = np.zeros((np.max(self.ctx.cells)+1, self.ctx.num_dimensions), dtype=np.float32)

        # Set up tp generator
        tp_gene = self.ctx.TDS.generate_tp_block()

        # Set up block counter (also dimension counter).
        block_count = 0

        # Each tp block is (num_vectors, 1) of np.float32. One block = all values for 1 dimension.
        for block in tp_gene:

            # Sort the block. N.B., can change np sort algorithm (https://numpy.org/doc/stable/reference/generated/numpy.sort.html)
            sorted_block = np.sort(block, axis=0)

            # Set first boundary_val (0) along current dimension to just less than min value
            self.ctx.boundary_vals[0, block_count] = sorted_block[0] - 0.001

            cells_for_dim = self.ctx.cells[block_count]

            # Loop over the number of cells allocated to current dimension - careful with indices, should start at 1 and go to penultimate.
            # If cells_for_dim = 32, this will go to idx 31. That's fine, because boundary_vals goes up to max(cells) + 1.
            for j in range(1, cells_for_dim):

                # Using math ceil; alternative is np
                self.ctx.boundary_vals[j, block_count] = sorted_block[
                    math.ceil(j * self.ctx.num_vectors / cells_for_dim)]

            # Set final boundary val along current dim
            # Using idx cells_for_dim is safe since boundary_vals goes up to max(cells) + 1
            self.ctx.boundary_vals[cells_for_dim, block_count] = sorted_block[self.ctx.num_vectors - 1] + 0.001

            # Increment block_count
            block_count += 1

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Operates on transposed file
    def _design_boundaries(self):
    
        # Set up tp generator
        tp_gene = self.ctx.TDS.generate_tp_block()

        # Set up block counter, this is also dimension counter
        block_count = 0

        # Loop over blocks (i.e. dimensions). Each block is (num_vectors, 1)
        for block in tp_gene:

            # Extract cells for current dimension
            cells_for_dim = self.ctx.cells[block_count]

            # If current dimension only has 1 cell (i.e. 0 bits allocated to it), then break and end.
            # Think values in self.cells are implicitly sorted descending.
            if self.ctx.cells[block_count] == 1:
                break

            # Call Lloyd's algorithm function -> could be replaced by modified Lloyds
            # MATLAB has B(1:CELLS(i)+1, i). Say a dim has 4 cells, this goes from 1 to 5, inclusive.
            # Ours will go from 0 to 5, not inclusive at the top, so really 0,1,2,3,4. Therefore equivalent. 
            if self.ctx.use_qhist:
                # r, c = self._weighted_lloyd(block, self.ctx.boundary_vals[0:cells_for_dim+1, block_count])
                r, c = self._lloyd(block, self.ctx.boundary_vals[0:cells_for_dim+1, block_count])
            else:
                r, c = self._lloyd(block, self.ctx.boundary_vals[0:cells_for_dim+1, block_count])

            self.ctx.boundary_vals[0:cells_for_dim+1, block_count] = c
            # self.ctx.boundary_vals[0:cells_for_dim, block_count] = c

            block_count += 1

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
            for j in range(1, num_boundary_vals):  # MATLAB has a -1 here... don't think we need?   

                c[j] = (r[j - 1] + r[j]) / 2

            # Stopping condition check
            # print("((delta - delta_new)/delta) -> ", ((delta - delta_new)/delta))
            if ((delta - delta_new) / delta) < stop:
                # print("Number of Lloyd's iterations: ", str(num_lloyd_iterations))
                return r, c
            delta = delta_new

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _weighted_lloyd(self, block, boundary_vals_in):

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

        print("self.weights: ", self.weights)

        num_lloyd_iterations = 0
        while True:
            delta_new = np.float32(0)
            num_lloyd_iterations += 1

            # Loop over intervals; careful with indices
            for i in range(num_boundary_vals - 1):

                # print("    i loop : ", i)                

                # Find values in block between boundary values; np.where?
                idxs = np.where(np.logical_and(block >= c[i], block < c[i + 1]))[0]

                X_i = block[idxs]
                W_i = self.weights[idxs]

                # If any values found
                if np.shape(X_i)[0] > 0:
                    r[i] = np.divide(np.sum(np.multiply(W_i, X_i)), np.sum(W_i))
                else:
                    r[i] = np.random.rand(1) * (M2 - M1) + M1

                # Add representation error over current interval to delta_new
                delta_new += np.sum(np.square(np.multiply(W_i, X_i) - r[i]))

            # Sort representative values - todo: sorting algorithm selection
            r = np.sort(r)

            # Update boundary values based on representative values
            for j in range(1, num_boundary_vals):  # MATLAB has a -1 here... don't think we need?   

                c[j] = (r[j - 1] + r[j]) / 2

            # Stopping condition check
            # print("((delta - delta_new)/delta) -> ", ((delta - delta_new)/delta))
            if ((delta - delta_new) / delta) < stop:
                print("Number of Lloyd's iterations: ", str(num_lloyd_iterations))
                return r, c

            delta = delta_new
    
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Operates on the transposed datafile. 
    def _create_vaqfile(self):

        # Could init this in loop?
        self.cset = np.ones(self.ctx.num_vectors, dtype=np.uint8)

        # Setup tp generator
        tp_gene = self.ctx.TDS.generate_tp_block()

        # Setup tp block counter -> also dimension counter
        block_count = 0

        # Open vaqfile -> write handle
        self._open_file('vaq', 'wb')

        # Loop over tp blocks (i.e. loop over dimensions)
        for block in tp_gene:

            # print("_create_vaqfile -> Dimension : ", block_count)

            # Loop over i in range(cells[block_count])
            for i in range(self.ctx.cells[block_count]):

                l = self.ctx.boundary_vals[i, block_count]
                r = self.ctx.boundary_vals[i + 1, block_count]
                A = np.where(np.logical_and(block >= l, block < r))[0]

                # MATLAB: Set CSET of those indices to the k-1. Effectively, if a record lies between the 1st and 2nd boundary value, assign it 
                # to the 0th cells (as this is really the cell bounded by boundary values 1 and 2.)
                # Python: Set it to k, rather than k-1. If it lies between boundary values 0 and 1, put it in cell 0.
                self.cset[A] = i

            # Write CSET to vaq_handle_write
            self.vaq_handle_write.write(self.cset)

            block_count += 1

        print(self.ctx.boundary_vals[0,:])
        
        # Close vaqfile
        self._close_file('vaq', 'wb')
    
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Gives (num_vectors, 1) block of vaq_index; all data for a single dimension.    
    def generate_vaq_block(self, start_offset=0):

        block_idx = start_offset

        # Reading a column of VAQ index per block. Each word (usually 4 bytes) contains 4 VAQ cells
        with open(self.full_vaq_fname, mode="rb") as f:

            while True:
                f.seek(self.ctx.tp_num_words_per_block * block_idx, os.SEEK_SET)
                block = np.fromfile(file=f, count=self.ctx.tp_num_words_per_block, dtype=np.uint8)

                if block.size > 0:
                    block = np.reshape(block, (self.ctx.num_vectors))
                    yield block
                    block_idx += 1
                else:
                    break

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Gives (num_vectors, 1) block of vaq_index; all data for a single dimension.    
    def generate_vaq_block_mem(self, start_offset=0):

        block_idx = start_offset

        # Reading a column (dimension) of VAQ index from in-memory self self.vaqdata. 
        while block_idx < self.ctx.num_dimensions:
            
            # yield self.vaqdata[:,block_idx].reshape((self.ctx.num_vectors, 1))
            yield self.vaqdata[:,block_idx].reshape(self.ctx.num_vectors,1)
            block_idx += 1
    
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _save_vaq_vars(self):
        np.savez(os.path.join(self.ctx.path, '') + self.ctx.fname + '.vaqvars', 
                 CELLS=self.ctx.cells, BOUNDARY_VALS=self.ctx.boundary_vals)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_vaq_vars(self):
        vaq_full_varfile = os.path.join(self.ctx.path, '') + self._find_file_by_suffix('.vaqvars.npz')

        print("Loading vaq variables from ", vaq_full_varfile)
        with np.load(vaq_full_varfile) as data:
            self.ctx.cells = data['CELLS']
            self.ctx.boundary_vals = data['BOUNDARY_VALS']

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _find_file_by_suffix(self, suffix):
        hit_count = 0
        hits = []
        for file in os.listdir(self.ctx.path):
            if file.endswith(suffix):
                hits.append(file)
                hit_count += 1
        if hit_count > 1:
            raise ValueError("Too many hits for suffix ", str(suffix))
        else:
            return hits[0]
    
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_vaqdata(self):
        data = np.fromfile(file=self.full_vaq_fname, count=-1, dtype=np.uint8)
        self.vaqdata = np.reshape(data,(self.ctx.num_vectors, self.ctx.num_dimensions), order="F")
        print()
        print("IN-MEMORY VAQ PROCESSING SELECTED!")
        print()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def build(self):
        
        # Calculate energies
        self._calc_energies()

        # Bit allocation (covers uniform + non-uniform)
        self._allocate_bits()

        # Assign weights (only if using qhist)
        if self.ctx.use_qhist:
            self._assign_weights()

        # Init boundary values
        self._init_boundaries()

        # Design boundary values (with Lloyd's)
        if self.ctx.design_boundaries:
            self._design_boundaries()

        # Save cells and boundary_vals for use elsewhere
        self._save_vaq_vars()

        # Create vaqfile
        self._create_vaqfile()

        # For mode F, if in-memory VAQ requested, load it
        if (self.ctx.mode == 'F') and (self.ctx.inmem_vaqdata):
            self._load_vaqdata()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def load(self):
        pass

    # #----------------------------------------------------------------------------------------------------------------------------------------
    # def vaq_generate_column(self):
    #     pass

    # #----------------------------------------------------------------------------------------------------------------------------------------
    # def vaq_generate_column_memory(self):
    #     pass
