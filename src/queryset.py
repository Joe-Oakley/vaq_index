import numpy as np
import os
import timeit

from qsession import QSession

class QuerySet:

    MAX_UINT8 = 255

    def __init__(self, ctx: QSession = None):
        self.ctx                    = ctx
        self.full_query_fname       = None
        self.full_res_fname         = None 
        self.full_metrics_fname     = None

        # Can't initialize these until _open_query_file()
        self.Q              = None # Query set
        self.num_queries    = None 
        self.first_stage    = None
        self.second_stage   = None 

        # A single transformed query vector.
        self.q = None

        # Re-initialized at the start of phase 1 for each query.
        self.ANS    = None
        self.UP     = None
        self.V      = None
        self.L      = None
        self.U      = None
        self.S1     = None
        self.S2     = None

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialize(self):

        # Set up filenames. If user has specified a query file name, use this, otherwise default to the dataset name as the stub
        if self.ctx.query_fname is not None:
            self.full_query_fname = os.path.join(self.ctx.path, '') + self.ctx.query_fname
            self.full_res_fname = os.path.join(self.ctx.path, '') + self.ctx.query_fname + '.res'
            self.full_metrics_fname = os.path.join(self.ctx.path, '') + self.ctx.query_fname + '.met'
        else:
            self.full_query_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '_qry'
            self.full_res_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.res'
            self.full_metrics_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.met'

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Build will call this function followed by _transform_query_file()
    def _open_query_file(self):

        # print("****************************************")
        # print("In _open_query_file()")
        # print("****************************************")

        # Read all queries into memory at once. Don't know how many; use -1 to get number of queries.
        # self._open_file('qry', 'rb')
        with open(self.full_query_fname, mode='rb') as f:

            # Should have run change_ends.py before this, so endianness shouldn't be a factor.
            if self.ctx.big_endian:
                # queries = np.fromfile(file=self.query_handle_read, count=-1, dtype=np.float32).byteswap(inplace=True)
                queries = np.fromfile(file=f, count=-1, dtype=np.float32).byteswap(inplace=True)
            else:
                queries = np.fromfile(file=f, count=-1, dtype=np.float32)

            # Reshape and trim identifiers
            print("queries shape before reshape + trim identifiers: ", np.shape(queries))
            queries = np.reshape(queries, (-1, self.ctx.num_dimensions+1), order="C")
            queries = np.delete(queries, 0, 1)
            print("queries shape after reshape + trim identifiers: ", np.shape(queries))
            
            # Set up remaining variables
            self.Q = queries
            self.num_queries = np.shape(queries)[0]
            self.first_stage = np.zeros((self.num_queries), dtype=np.uint32)
            self.second_stage = np.zeros((self.num_queries), dtype=np.uint32)        

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Perform transform on self.Q: Q=(Q-repmat(DATA_MEAN',QUERY_FILE_SIZE,1))*KLT';
    def _transform_query_file(self):

        # print("****************************************")
        # print("In _transform_query_file()")
        # print("****************************************")
        
        dim0_pretransform = self.Q[:,0].copy()
        q0_pretransform = self.Q[0,:].copy()
        
        rep_mean = np.tile(self.ctx.dim_means, (self.num_queries,1)) # (num_queries, num_dims)
        X = self.Q # (num_queries, num_dims) 
        Y = np.subtract(X, rep_mean)
        Z = np.matmul(Y,self.ctx.transform_matrix)
        self.Q = Z

        dim0_posttransform = self.Q[:,0].copy()
        q0_posttransform = self.Q[0,:].copy()

        print()
        print("First dimension BEFORE transform   : ", dim0_pretransform.shape, dim0_pretransform)  # This is (1,50) -> The first dimension value across all the queries
        print("First dimension AFTER  transform   : ", dim0_posttransform.shape, dim0_posttransform)  # This is (1,50) -> The first dimension value across all the queries        
        print()
        print("First query point BEFORE transform : ", q0_pretransform.shape, q0_pretransform)  # This is (1,50) -> The first dimension value across all the queries
        print("First query point AFTER  transform : ", q0_posttransform.shape, q0_posttransform)  # This is (1,50) -> The first dimension value across all the queries        

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _write_qhist_info(self, mode, query_idx=None): # File size (bytes) = word_size + num_queries(4*word_size + word_size(2*query_k))

        # Open file
        # self._open_file('qhist', 'ab')
        with open(self.ctx.qhist_fname, mode='ab') as f:

            if mode == 'start':
                # self.qhist_handle_append.write(np.uint32(self.num_queries))
                f.write(np.uint32(self.num_queries))
                print("np.uint32(self.num_queries): ", str(np.uint32(self.num_queries)))
            elif mode == 'main':
                # Write Query ID, query_k, phase 1 elims, phase 2 visits
                # self.qhist_handle_append.write(np.uint32(query_idx))
                # self.qhist_handle_append.write(np.uint32(self.ctx.query_k))
                # self.qhist_handle_append.write(np.uint32(self.first_stage[query_idx]))
                # self.qhist_handle_append.write(np.uint32(self.second_stage[query_idx]))
                f.write(np.uint32(query_idx))
                f.write(np.uint32(self.ctx.query_k))
                f.write(np.uint32(self.first_stage[query_idx]))
                f.write(np.uint32(self.second_stage[query_idx]))

                # Write k pairs of (NN vector ID, Euclidean distance to query point)
                for j in range(self.ctx.query_k):
                    # self.qhist_handle_append.write(np.float32(self.V[j])) # Really an int, but using float for ease of reading back in.
                    # self.qhist_handle_append.write(np.float32(self.ANS[j]))
                    f.write(np.float32(self.V[j])) # Really an int, but using float for ease of reading back in.
                    f.write(np.float32(self.ANS[j]))
            else:
                raise ValueError("Invalid mode selected: ", mode)

        # Close file
        # self._close_file('qhist', 'ab')

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Res file size (bytes) = word_size * 2 (i.e. V, ANS) * self.ctx.query_k * self.num_queries.
    def _write_res_info(self): # Called once per query. 
    
        # Open res file (mode append)
        # self._open_file('res', 'ab')
        with open(self.full_res_fname, mode='ab') as f:

            # Write self.V and self.ANS (refreshed per-query.)
            for i in range(self.ctx.query_k):
                f.write(np.uint32(self.V[i])) # uint32
                f.write(np.float32(self.ANS[i])) # float32

        # Close res file
        # self._close_file(self.res_handle_append)
        # self._close_file('res', 'ab')

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _print_res_info(self, query_idx):

        # Display results
        print()
        print("********************")
        print("Results for Query ", query_idx)
        print("********************")
        print("V")
        print(self.V)
        print("ANS")
        print(self.ANS)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _write_metrics_info(self): # Write metrics info for all queries.

        # Open metrics file
        # self._open_file('met', 'wb')
        with open(self.full_metrics_fname, 'wb') as f:

            # Write first stage and second stage
            f.write(self.first_stage) # (num_queries, 1), uint32
            f.write(self.second_stage) # (num_queries, 1), uint32

        # Close metrics file
        # self._close_file('met', 'wb')

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _print_metrics_info(self):

        print()
        print("Overall Results for Run")
        print("First Stage : ")
        print(self.first_stage)
        print("Second Stage : ")
        print(self.second_stage)
        print()
        print(" <---- END OF VAQPLUS PROCESSING RUN ---->")
        print()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # This function will be called repeatedly in a query loop from another function.
    def _run_phase_one(self, query_idx):

        print()
        print("****************************************")
        print("Now processing query idx: ", str(query_idx))
        print("****************************************")

        self.q = self.Q[query_idx,:] # self.Q is (num_queries,num_dims) -> self.q is (num_dims,) -> The first query
        # print("shape of self.q at start of run_phase_one: ", str(np.shape(self.q)))

        # These are (1, 50) for 50-NN 
        self.ANS = np.ones((self.ctx.query_k), dtype=np.float32)*np.inf
        self.UP = np.ones((self.ctx.query_k), dtype=np.float32)*np.inf
        self.V = np.ones((self.ctx.query_k), dtype=np.uint32)*np.inf

        self.L = np.zeros((self.ctx.num_vectors), dtype=np.float32)
        self.U = np.zeros((self.ctx.num_vectors), dtype=np.float32)
        self.S1 = np.zeros((self.ctx.num_vectors), dtype=np.float32)
        self.S2 = np.zeros((self.ctx.num_vectors), dtype=np.float32)

        # Set up VAQIndex generator (one block = one dimension)
        if self.ctx.inmem_vaqdata:
            vaq_gene = self.ctx.VAQ.generate_vaq_block_mem()
        else:
            vaq_gene = self.ctx.VAQ.generate_vaq_block()
        
        num_vectors_per_block = self.ctx.num_vectors_per_block
        num_blocks = self.ctx.num_blocks

        # 21/10/2023    Create a boundary-distances arrays containing squared lower/upper bound distances from the query vector to 
        #               each of the non-zero boundary_vals. This will be used below to avoid repeatedly calculating these distances 
        #               when comparing the query to the vectors
        calcdists_reft = timeit.default_timer()
        
        boundary_vals_wrapped = np.roll(self.ctx.boundary_vals,-1,0) # Roll rows; 1st row becomes 0th, 0th becomes last.
        D1 = np.abs(np.subtract(boundary_vals_wrapped, self.q[:,None].ravel(), where=boundary_vals_wrapped!=0, out=np.zeros_like(boundary_vals_wrapped) )  ) # cset+1
        D2 = np.abs(np.subtract(self.ctx.boundary_vals, self.q[:,None].ravel(), where=self.ctx.boundary_vals!=0, out=np.zeros_like(self.ctx.boundary_vals) )  ) # cset
        D_MIN = np.square(np.minimum(D1,D2))
        D_MAX = np.square(np.maximum(D1,D2))
        
        msg = 'Query : ' + str(query_idx) + ' Calc boundary LB/UB distances'
        self.ctx.debug_timer('QuerySet._run_phase_one',calcdists_reft, msg, 1)
        
        dimloop_reft = timeit.default_timer()

        block_count = 0 # MATLAB j
        for CSET in vaq_gene: # cset is a block of VAQ -> (num_vectors, 1)

            cells_for_dim = self.ctx.cells[block_count]
            qj = self.q[block_count]

            # The np.where selects RIGHT boundaries, but target_cells will be set to values 1 lower than the corresponding boundary_vals,
            # because we're only searching over [1:cells_for_dim+1] i.e. starting at row 1. So these are really LEFT boundaries.
            target_cells = np.where(qj <= self.ctx.boundary_vals[1:cells_for_dim+1, block_count]) 
            
            if target_cells[0].size == 0:
                R = cells_for_dim # If qj > all boundary_vals, put in final cell -> this is the LEFT boundary of final cell.
            else:
                R = np.min(target_cells[0]) 

            # Populate chunks of self.S1 and self.S2 (could probably remove this block loop)
            for i in range(num_blocks): 

                qblock_reft = timeit.default_timer()

                # cset should be (num_vectors_per_block,1)
                cset = CSET[i*num_vectors_per_block: (i+1)*num_vectors_per_block]         

                # Use cset values to index into the distances array created for the query
                self.S1[i*num_vectors_per_block: (i+1)*num_vectors_per_block] = D_MIN[cset,block_count]
                self.S2[i*num_vectors_per_block: (i+1)*num_vectors_per_block] = D_MAX[cset,block_count]

                # msg = 'Query ' + str(query_idx) + ' Dim ' + str(block_count) + ' Chunk ' + str(i) + ' Populating S1/S2 chunk'
                # self.ctx.debug_timer('QuerySet._run_phase_one', qblock_reft, msg, 2)

            # x = np.logical_not(CSET == R).astype(np.int32)
            x = np.logical_not(CSET == R).astype(np.float32)

            # Calculate L (lower bound): L=L+x.*S1;    L is (num_vectors, 1).
            # Adds the lower bound distance for the dimension in question, to a running total which becomes the overall (squared) lower bound distance. 
            # Mask x ensures that points in the same interval along a given dimension have LB distance 0 over that dimension.
            self.L = self.L + np.multiply(x, self.S1)

            # Calculate U (upper bound)
            self.U = self.U + self.S2

            # Increment block counter
            block_count += 1

        # End block loop

        msg = 'Query : ' + str(query_idx) + ' Dimensions block'        
        self.ctx.debug_timer('QuerySet._run_phase_one',dimloop_reft, msg, 1)

        elimcount_reft = timeit.default_timer()

        # Elim counting
        elim = 0

        get_max_next_time = True
        for i in range(self.ctx.num_vectors):

            if get_max_next_time:
                max_up = self.UP.max() # https://stackoverflow.com/questions/10943088/numpy-max-or-max-which-one-is-faster
                max_up_idx = np.argmax(self.UP) # If multiple same max val, returns min idx.
            if self.L[i] <= max_up:
                if self.U[i] <= max_up:
                    self.UP[max_up_idx] = self.U[i]
                    get_max_next_time = True
                else:
                    get_max_next_time = False
                    
                    # # Adding elim += 1 makes P1 elims inconsistent with P2 visits.
                    # elim += 1 

            else:
                elim += 1

        self.UP = np.sort(self.UP, axis=0) # Not needed?

        self.first_stage[query_idx] = elim

        msg = 'Query : ' + str(query_idx) + ' Elim Counting'        
        self.ctx.debug_timer('QuerySet._run_phase_one',elimcount_reft, msg, 1)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # May have a separate version of this function for the case where the entire VAQIndex fits into memory.
    # This function will be called repeatedly in a query loop from another function.
    def _run_phase_one_inmem(self, query_idx):

        print()
        print("****************************************")
        print("Now processing query idx: ", str(query_idx))
        print("****************************************")

        print("shape of self.Q: ", str(np.shape(self.Q)))
        self.q = self.Q[query_idx,:] # self.Q is (num_queries,num_dims) -> self.q is (num_dims,) -> The first query
        print("shape of self.q at start of run_phase_one: ", str(np.shape(self.q)))

        # These are (1, 50) for 50-NN 
        self.ANS = np.ones((self.ctx.query_k), dtype=np.float32)*np.inf
        self.UP = np.ones((self.ctx.query_k), dtype=np.float32)*np.inf
        self.V = np.ones((self.ctx.query_k), dtype=np.uint32)*np.inf

        # 23/10/2033
        self.L = np.zeros((self.ctx.num_vectors), dtype=np.float32)
        self.U = np.zeros((self.ctx.num_vectors), dtype=np.float32)        

        # 21/10/2023    Create a boundary-distances arrays containing squared upper/lower bound distances from the query vector to 
        #               each of the non-zero boundary_vals. This will be used below to avoid repeatedly calculating these distances 
        #               when comparing the query to the vectors
        calcdists_reft = timeit.default_timer()
        
        boundary_vals_wrapped = np.roll(self.ctx.boundary_vals,-1,0)
        D1 = np.abs(np.subtract(boundary_vals_wrapped, self.q[:,None].ravel(), where=boundary_vals_wrapped!=0, out=np.zeros_like(boundary_vals_wrapped) )  )
        D2 = np.abs(np.subtract(self.ctx.boundary_vals, self.q[:,None].ravel(), where=self.ctx.boundary_vals!=0, out=np.zeros_like(self.ctx.boundary_vals) )  )
        D_MIN = np.square(np.minimum(D1,D2)) # Same dims as self.ctx.boundary_vals.
        D_MAX = np.square(np.maximum(D1,D2)) # Same dims as self.ctx.boundary_vals.
        
        msg = 'Query : ' + str(query_idx) + ' Calc boundary LB/UB distances'
        self.ctx.debug_timer('QuerySet._run_phase_one',calcdists_reft, msg, 1)

        # 22.10.2023: Produce a (128,) array containing index positions of selected boundary value (LEFT boundaries) for each dimension
        min_target_cells = np.min( np.where( ( self.q <= self.ctx.boundary_vals ) & ( self.ctx.boundary_vals != 0), self.ctx.boundary_vals, np.inf) , axis=0)

        R_cells = np.zeros(self.ctx.num_dimensions, dtype=np.uint8)
        for i in range(self.ctx.num_dimensions):
            
            # Ensures consistency with _run_phase_one for very large values.
            if min_target_cells[i] == np.inf:
                R_cells[i] = self.ctx.cells[i]
            else:
                R_cells[i] = np.where(min_target_cells[i] == self.ctx.boundary_vals[1:self.ctx.cells[i]+1,i])[0]   
        
        num_vectors_per_block = self.ctx.num_vectors_per_block
        dimloop_reft = timeit.default_timer()

        for blockno in range(self.ctx.num_blocks): 
            
            qblock_reft = timeit.default_timer()
            
            # vaqdata is (num_vectors, num_dimensions)
            RSET = self.ctx.VAQ.vaqdata[(blockno*num_vectors_per_block):((blockno+1)*num_vectors_per_block), :]

            # x = np.logical_not(RSET == R_cells).astype(np.int32)
            x = np.logical_not(RSET == R_cells).astype(np.float32)    
            
            # 23/10/2023    Use RSET values to index into the distances array created for the query
            dmin = np.take_along_axis(D_MIN, RSET, axis=0) # Uses RSET values to index into D_MIN. dmin same shape as RSET, but contains distances.
            dmax = np.take_along_axis(D_MAX, RSET, axis=0)
    
            lbounds = np.multiply(x, dmin)
            ubounds = dmax
            
            self.L[blockno*self.ctx.num_vectors_per_block: (blockno+1)*self.ctx.num_vectors_per_block] = np.sum(lbounds,axis=1)
            self.U[blockno*self.ctx.num_vectors_per_block: (blockno+1)*self.ctx.num_vectors_per_block] = np.sum(ubounds,axis=1)

            msg = 'Query ' + str(query_idx) + ' Dim ' + str(i) + ' Chunk ' + str(i) + ' Populating S1/S2 chunk'
            self.ctx.debug_timer('QuerySet._run_phase_one', qblock_reft, msg, 2)
                
        msg = 'Query : ' + str(query_idx) + ' Dimensions block'        
        self.ctx.debug_timer('QuerySet._run_phase_one',dimloop_reft, msg, 1)

        elimcount_reft = timeit.default_timer()

        # Elim counting
        elim = 0
        get_max_next_time = True
        for i in range(self.ctx.num_vectors):

            if get_max_next_time:
                max_up = self.UP.max()
                max_up_idx = np.argmax(self.UP) # If multiple same max val, returns min idx.
                
            if self.L[i] <= max_up:
                if self.U[i] <= max_up:
                    self.UP[max_up_idx] = self.U[i]
                    get_max_next_time = True
                else:
                    get_max_next_time = False
            else:
                    elim += 1
            
        self.UP = np.sort(self.UP, axis=0) # Not needed?    
        self.first_stage[query_idx] = elim

        msg = 'Query : ' + str(query_idx) + ' Elim Counting'        
        self.ctx.debug_timer('QuerySet._run_phase_one',elimcount_reft, msg, 1)
    
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _run_phase_two(self, query_idx):
        
        # Sort L (lower bounds) into [LL, J], also need original indices 
        J = np.argsort(self.L,axis=0)
        LL = np.sort(self.L,axis=0)

        # Each random read on transformed file should seek to start of the info for current record, then read num_dimensions words.
        num_words_random_read = self.ctx.num_dimensions

        # Loop over all vectors; is this sensible; don't we just want to consider candidates only in terms of their LBs?
        vectors_considered_p2 = 0

        for i in range(self.ctx.num_vectors):

            # If lower bound of i is greater than 50th best upper bound, stop
            if LL[i] > self.ANS[self.ctx.query_k -1]:
                break
            else:
                # Random read (of num_dimensions words) from transformed file. 
                start_offset = J[i]*self.ctx.num_dimensions*self.ctx.word_size
                TSET = self.ctx.TDS.tf_random_read(start_offset, num_words_random_read) # (1, num_dimensions)

                # Append squared distance between self.q (could also use self.Q[query_idx, :]) and the vector read from disk to ANS
                # self.q and TSET are both (1, num_dimensions)
                T = np.append(self.ANS, np.sum(np.square(self.q - TSET)))

                # Append J[i] to V
                W = np.append(self.V, J[i])

                # Sort ANS and also keep original locations
                I = np.argsort(T)
                T = np.sort(T)

                # Trim ANS to only first query_k
                self.ANS = T[0:self.ctx.query_k]

                # V=W(I(1:QUERY_SIZE)); think these are the indices of answers?
                self.V = W[I[0:self.ctx.query_k]]

                # Increment counter; not using i since we'll lose it after the loop
                vectors_considered_p2 += 1
        
        # Done with search

        self.second_stage[query_idx] = vectors_considered_p2

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _run_queries(self):
        
        # If creating qhist file, write num_queries to the start of the file.
        if self.ctx.create_qhist:
            self._write_qhist_info(mode='start')

        # Loop over all queries
        for i in range(self.num_queries):

            # Phase one
            reft = timeit.default_timer()

            if self.ctx.inmem_vaqdata:
                # self._run_phase_one_inmem(i) # Full in-mem
                self._run_phase_one(i) # Partial in-mem
            else:
                self._run_phase_one(i)

            msg = 'Query : ' + str(i) + ' Phase 1 duration'
            self.ctx.debug_timer('QuerySet._run_queries',reft, msg)
            
            # Phase two
            reft = timeit.default_timer()
            self._run_phase_two(i)
            msg = 'Query : ' + str(i) + ' Phase 2 duration'            
            self.ctx.debug_timer('QuerySet._run_queries', reft, msg)

            # Save self.V and self.ANS for current query
            self._write_res_info() # Mode 'ab'

            # Print results of current query
            self._print_res_info(i)

            # If creating qhist, write Query ID, query_k, phase 1 elims, phase 2 visits for current query
            if self.ctx.create_qhist:
                self._write_qhist_info(mode='main', query_idx=i)
        
        # Save overall query metrics (all queries)
        self._write_metrics_info()

        # Print overall metrics for query run (all queries)
        self._print_metrics_info()
        
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def process(self):

        # Initialize
        self._initialize()

        # Open query file
        self._open_query_file()

        # Transform query file
        self._transform_query_file()

        # Run queries
        self._run_queries()