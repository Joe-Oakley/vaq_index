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
        # self.full_qhist_fname       = None

        self.query_handle_read      = None
        self.query_handle_write     = None
        self.query_handle_append    = None
        self.res_handle_read        = None
        self.res_handle_write       = None
        self.res_handle_append      = None
        self.metrics_handle_read    = None
        self.metrics_handle_write   = None
        self.metrics_handle_append  = None
        self.qhist_handle_read      = None
        self.qhist_handle_write     = None
        self.qhist_handle_append    = None

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
    def _open_file(self, ftype, mode):

        if ftype == 'qry':
            if mode == 'rb':
                self.query_handle_read = open(self.full_query_fname, mode=mode)
            elif mode == 'wb':
                self.query_handle_write = open(self.full_query_fname, mode=mode)
            elif mode == 'ab':
                self.query_handle_append = open(self.full_query_fname, mode=mode)
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'res':
            if mode == 'rb':
                self.res_handle_read = open(self.full_res_fname, mode=mode)
            elif mode == 'wb':
                self.res_handle_write = open(self.full_res_fname, mode=mode)
            elif mode == 'ab':
                self.res_handle_append = open(self.full_res_fname, mode=mode)
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'met':
            if mode == 'rb':
                self.metrics_handle_read = open(self.full_metrics_fname, mode=mode)
            elif mode == 'wb':
                self.metrics_handle_write = open(self.full_metrics_fname, mode=mode)
            elif mode == 'ab':
                self.metrics_handle_append = open(self.full_metrics_fname, mode=mode)    
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'qhist':
            if mode == 'rb':
                self.qhist_handle_read = open(self.ctx.qhist_fname, mode=mode)
            elif mode == 'wb':
                self.qhist_handle_write = open(self.ctx.qhist_fname, mode=mode)
            elif mode == 'ab':
                self.qhist_handle_append = open(self.ctx.qhist_fname, mode=mode)
            else:
                raise ValueError("Invalid mode selected: ", mode)
        else:
            raise ValueError("Invalid ftype selected: ", ftype)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self, ftype, mode): # ftypes: qry, res, met, qhist. modes: 'rb', 'wb', 'ab'

        if ftype == 'qry':
            if mode == 'rb':
                self.query_handle_read.close()
                self.query_handle_read = None
            elif mode == 'wb':
                self.query_handle_write.close()
                self.query_handle_write = None
            elif mode == 'ab':
                self.query_handle_append.close()
                self.query_handle_append = None
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'res':
            if mode == 'rb':
                self.res_handle_read.close()
                self.res_handle_read = None
            elif mode == 'wb':
                self.res_handle_write.close()
                self.res_handle_write = None
            elif mode == 'ab':
                self.res_handle_append.close()
                self.res_handle_append = None
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'met':
            if mode == 'rb':
                self.metrics_handle_read.close()
                self.metrics_handle_read = None
            elif mode == 'wb':
                self.metrics_handle_write.close()
                self.metrics_handle_write = None
            elif mode == 'ab':
                self.metrics_handle_append.close()
                self.metrics_handle_append = None
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'qhist':
            if mode == 'rb':
                self.qhist_handle_read.close()
                self.qhist_handle_read = None
            elif mode == 'wb':
                self.qhist_handle_write.close()
                self.qhist_handle_write = None
            elif mode == 'ab':
                self.qhist_handle_append.close()
                self.qhist_handle_append = None
            else:
                raise ValueError("Invalid mode selected: ", mode)
        else:
            raise ValueError("Invalid ftype selected: ", ftype)

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
    # Open query read handle -> read entire query file into memory, do byteswap if needed -> reshape to determine number of queries ->
    # set variables. Build will call this function followed by _transform_query_file()
    # 6Oct; Don't need to worry about endianness, but they will have IDs, so make sure we strip.
    def _open_query_file(self):

        print("****************************************")
        print("In _open_query_file()")
        print("****************************************")

        # Read all queries into memory at once. Don't know how many; use -1 to get number of queries.
        self._open_file('qry', 'rb')

        if self.ctx.big_endian:
            queries = np.fromfile(file=self.query_handle_read, count=-1, dtype=np.float32).byteswap(inplace=True)
        else:
            queries = np.fromfile(file=self.query_handle_read, count=-1, dtype=np.float32)

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
    def _transform_query_file(self):

        print("****************************************")
        print("In _transform_query_file()")
        print("****************************************")

        # Perform transform on self.Q: Q=(Q-repmat(DATA_MEAN',QUERY_FILE_SIZE,1))*KLT';
        # Q is (num_queries, num+dims). dim_means is (num_dims, 1), so transposed is (1,num_dims)
        # repmat copies a (1,num_dims) across num_queries rows and 1 column -> makes it (num_queries, num_dims) to match Q.
        # multiply by KLT', which is still (num_dims,num_dims)
        # repmat code example: rep_mean = np.tile(self.dim_means, (1, self.num_vectors_per_block))
        dim0_pretransform = self.Q[:,0].copy()
        q0_pretransform = self.Q[0,:].copy()

        # The commented line + block below both aim to apply the transformation to the queries. The commented line is an exact match to 
        # the MATLAB code. The block below was written to mirror the transformation done to the main dataset in transformed.py (with an
        # additional transpose to Q). Not sure which is the best option, but the choice will have repercussions, e.g. L191 dimensions 
        # need to be swapped.

        # self.Q = np.matmul((self.Q - np.tile(self.dim_means.T, (self.num_queries, 1))), self.transform_matrix.T) # ORIGINAL
        
        rep_mean = np.tile(self.ctx.dim_means, (self.num_queries,1)) 
        X = self.Q  
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
        self._open_file('qhist', 'ab')

        if mode == 'start':
            self.qhist_handle_append.write(np.uint32(self.num_queries))
            print("np.uint32(self.num_queries): ", str(np.uint32(self.num_queries)))
        elif mode == 'main':
            # Write Query ID, query_k, phase 1 elims, phase 2 visits
            self.qhist_handle_append.write(np.uint32(query_idx))
            self.qhist_handle_append.write(np.uint32(self.ctx.query_k))
            self.qhist_handle_append.write(np.uint32(self.first_stage[query_idx]))
            self.qhist_handle_append.write(np.uint32(self.second_stage[query_idx]))

            # Write k pairs of (NN vector ID, Euclidean distance to query point)
            for j in range(self.ctx.query_k):
                self.qhist_handle_append.write(np.float32(self.V[j])) # Really an int, but using float for ease of reading back in.
                self.qhist_handle_append.write(np.float32(self.ANS[j]))
        else:
            raise ValueError("Invalid mode selected: ", mode)

        # Close file
        # self._close_file(self.qhist_handle_append)
        self._close_file('qhist', 'ab')

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _write_res_info(self): # Called once per query. 
    # Res file size (bytes) = word_size * 2 (i.e. V, ANS) * self.ctx.query_k * self.num_queries.
        
        # Open res file (mode append)
        self._open_file('res', 'ab')

        # Write self.V and self.ANS (refreshed per-query.)
        for i in range(self.ctx.query_k):
            self.res_handle_append.write(np.uint32(self.V[i])) # uint32
            self.res_handle_append.write(np.float32(self.ANS[i])) # float32

        # Close res file
        # self._close_file(self.res_handle_append)
        self._close_file('res', 'ab')

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
    def _write_metrics_info(self): # For all queries.
    # Metrics file size = 

        # Open metrics file
        self._open_file('met', 'wb')

        # Write first stage and second stage
        self.metrics_handle_write.write(self.first_stage) # (num_queries, 1), uint32
        self.metrics_handle_write.write(self.second_stage) # (num_queries, 1), uint32

        # Close metrics file
        # self._close_file(self.metrics_handle_write)
        self._close_file('met', 'wb')

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
    # May have a separate version of this function for the case where the entire VAQIndex fits into memory.
    # This function will be called repeatedly in a query loop from another function.
    def _run_phase_one(self, query_idx):

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

        # These are all (num_vectors, 1)
        # self.L = np.zeros((self.ctx.num_vectors,1), dtype=np.float32)
        # self.U = np.zeros((self.ctx.num_vectors,1), dtype=np.float32)
        # self.S1 = np.zeros((self.ctx.num_vectors,1), dtype=np.float32)
        # self.S2 = np.zeros((self.ctx.num_vectors,1), dtype=np.float32)
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
        
        boundary_vals_wrapped = np.roll(self.ctx.boundary_vals,-1,0)
        D1 = np.abs(np.subtract(boundary_vals_wrapped, self.q[:,None].ravel(), where=boundary_vals_wrapped!=0, out=np.zeros_like(boundary_vals_wrapped) )  )
        D2 = np.abs(np.subtract(self.ctx.boundary_vals, self.q[:,None].ravel(), where=self.ctx.boundary_vals!=0, out=np.zeros_like(self.ctx.boundary_vals) )  )
        D_MIN = np.square(np.minimum(D1,D2))
        D_MAX = np.square(np.maximum(D1,D2))
        
        msg = 'Query : ' + str(query_idx) + ' Calc boundary LB/UB distances'
        self.ctx.debug_timer('QuerySet._run_phase_one',calcdists_reft, msg, 1)
        
        dimloop_reft = timeit.default_timer()

        block_count = 0 # MATLAB j
        # Block/dimension loop
        for CSET in vaq_gene: # block = cset -> (num_vectors, 1)

            # print()    
            # print("Block/Dimension : ", block_count)
            # print("----------------------")

            cells_for_dim = self.ctx.cells[block_count]
            qj = self.q[block_count]

            # Calculate R
            # R=min(find(q(j)<=B(2:CELLS(j)+1,j)))-1;
            # Based on notes: We want to find the smallest boundary val index to the right of the query point's true value. We 
            # then subtract 1 from it. For this reason, we first look at boundary_vals[1], as this is the lowest boundary value
            # index which makes up the "right hand side" of a cell. After subtracting 1, this gives us the left hand side of the interval
            # which contains the query point. This is just a scalar.
            
            # R = np.min(np.where(qj <= self.boundary_vals[1:cells_for_dim+1, block_count])) - 1    # ORIG
            
            # Experiment 1
            target_cells = np.where(qj <= self.ctx.boundary_vals[1:cells_for_dim+1, block_count])   # Closest to MATLAB - seems best

            # target_cells = np.where(qj <= self.ctx.boundary_vals[0:cells_for_dim, block_count])   # What we have been doing

            # target_cells = np.where(qj <= self.ctx.boundary_vals[0:cells_for_dim+1, block_count])   # Possible
            
            
            if target_cells[0].size == 0:
                # R = cells_for_dim - 1
                R = cells_for_dim
            else:
                R = np.min(target_cells[0])
                # R = np.clip(np.min(target_cells[0]) - 1, 0, QuerySet.MAX_UINT8)    

            # Calculate q_rep (can't call it Q again!)
            # In MATLAB, we only repmat NO_OF_BLOCKS times. This means that the following happens (num_vectors/num_blocks) = num_vectors_per_block times.
            # Here, I'm tiling it num_vectors_per_block times to create a (num_vectors_per_block, 1). 
            # The loop below will thus only happen num_blocks times.
            qj_rep = np.tile(self.q[block_count], (num_vectors_per_block, 1))

            for i in range(num_blocks): 

                qblock_reft = timeit.default_timer()

                # cset should be (num_vectors_per_block,1)
                cset = CSET[i*num_vectors_per_block: (i+1)*num_vectors_per_block]         

                # 21/10/2023    Use cset values to index into the distances array created for the query
                self.S1[i*num_vectors_per_block: (i+1)*num_vectors_per_block] = D_MIN[cset,block_count]
                self.S2[i*num_vectors_per_block: (i+1)*num_vectors_per_block] = D_MAX[cset,block_count]

                # msg = 'Query ' + str(query_idx) + ' Dim ' + str(block_count) + ' Chunk ' + str(i) + ' Populating S1/S2 chunk'
                # self.ctx.debug_timer('QuerySet._run_phase_one', qblock_reft, msg, 2)

            x = np.logical_not(CSET == R).astype(np.int32)

            # Calculate L (lower bound): L=L+x.*S1;
            # L is (num_vectors, 1).
            # Adds the lower bound distance for the dimension in question, to a running total which becomes the overall (squared) lower bound distance. 
            # The reason we have to apply the mask x for this step is that if the query point and any of the data points reside in the same interval 
            # (along dimension j i.e. block_count), their lower bound distance needs to be 0.
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

        # Make a heap of self.UP (heapify)
        # UP_heap = np.multiply(self.UP, -1).tolist()
        # heapify(UP_heap)
        get_max_next_time = True
        for i in range(self.ctx.num_vectors):

            # # Original
            # if self.L[i] <= self.UP[self.ctx.query_k - 1]:
            #     UP_appended_sorted = np.sort(np.append(self.UP, self.U[i])) # Diff variable names than MATLAB
            #     self.UP = UP_appended_sorted[0:self.ctx.query_k]

            # New algo:
            # Check if U[i] is < np.max(self.UP) -> get the max in a var. If so, replace it.
            if get_max_next_time:
                max_up = self.UP.max() # https://stackoverflow.com/questions/10943088/numpy-max-or-max-which-one-is-faster
                max_up_idx = np.argmax(self.UP) # If multiple same max val, returns min idx.
            if self.L[i] <= max_up:
                if self.U[i] <= max_up:
                    self.UP[max_up_idx] = self.U[i]
                    get_max_next_time = True
                else:
                    # Adding elim += 1 here improves P1, but makes P2 reads inconsistent with P1 prunes.
                    get_max_next_time = False
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
        D_MIN = np.square(np.minimum(D1,D2))
        D_MAX = np.square(np.maximum(D1,D2))
        
        msg = 'Query : ' + str(query_idx) + ' Calc boundary LB/UB distances'
        self.ctx.debug_timer('QuerySet._run_phase_one',calcdists_reft, msg, 1)

        # Calculate R_cells
        # 22.10.2023: Objective is to produce a (128,) array containing index positions of selected boundary value for each dimension
        min_target_cells = np.min( np.where( ( self.q <= self.ctx.boundary_vals ) & ( self.ctx.boundary_vals != 0), self.ctx.boundary_vals, np.inf) , axis=0)
        final_min_target_cells = np.where(min_target_cells !=np.inf, min_target_cells, np.max(self.ctx.boundary_vals, axis=0))

        R_cells = np.zeros(self.ctx.num_dimensions, dtype=np.uint8)
        for i in range(self.ctx.num_dimensions):
            # R_cells[i] = np.squeeze( np.where(final_min_target_cells[i] == self.ctx.boundary_vals[0:self.ctx.cells[i]+1,i])[0]  )
            R_cells[i] = np.where(final_min_target_cells[i] == self.ctx.boundary_vals[0:self.ctx.cells[i]+1,i])[0] - 1        

            # # Debugging when investigating boundary selection
            # print(self.q)
            # print()
            # print(self.ctx.boundary_vals[0:self.ctx.cells[i]+1,i])
            # print(R_cells[i])
            # print()

        
        num_vectors_per_block = self.ctx.num_vectors_per_block
        dimloop_reft = timeit.default_timer()

        for blockno in range(self.ctx.num_blocks): 
            
            # print()    
            # print("Block : ", blockno)
            # print("-----------")
            
            qblock_reft = timeit.default_timer()
            
            RSET = self.ctx.VAQ.vaqdata[(blockno*num_vectors_per_block):((blockno+1)*num_vectors_per_block), :]
            x = np.logical_not(RSET == R_cells).astype(np.int32)    
            
            # 23/10/2023    Use RSET values to index into the distances array created for the query
            dmin = np.take_along_axis(D_MIN, RSET, axis=0)
            dmax = np.take_along_axis(D_MAX, RSET, axis=0)
    
            lbounds = np.multiply(x, dmin)
            ubounds = dmax
            
            self.L[blockno*self.ctx.num_vectors_per_block: (blockno+1)*self.ctx.num_vectors_per_block] = np.sum(lbounds,axis=1)
            self.U[blockno*self.ctx.num_vectors_per_block: (blockno+1)*self.ctx.num_vectors_per_block] = np.sum(ubounds,axis=1)

            msg = 'Query ' + str(query_idx) + ' Dim ' + str(i) + ' Chunk ' + str(i) + ' Populating S1/S2 chunk'
            self.ctx.debug_timer('QuerySet._run_phase_one', qblock_reft, msg, 2)


        # Calculate L (lower bound): L=L+x.*S1;
        # L is (num_vectors, 1).
        # Adds the lower bound distance for the dimension in question, to a running total which becomes the overall (squared) lower bound distance. 
        # The reason we have to apply the mask x for this step is that if the query point and any of the data points reside in the same interval 
        # (along dimension j i.e. block_count), their lower bound distance needs to be 0.
                
        msg = 'Query : ' + str(query_idx) + ' Dimensions block'        
        self.ctx.debug_timer('QuerySet._run_phase_one',dimloop_reft, msg, 1)

        elimcount_reft = timeit.default_timer()

        # Elim counting
        elim = 0
        get_max_next_time = True
        for i in range(self.ctx.num_vectors):
            
            
            # # ORIGINAL Algorithm (slow)                
            # if self.L[i] <= self.UP[self.ctx.query_k - 1]:
            #     UP_appended_sorted = np.sort(np.append(self.UP, self.U[i])) # Diff variable names than MATLAB
            #     self.UP = UP_appended_sorted[0:self.ctx.query_k]
            # else:
            #     elim += 1

            # NEW Algorithm (25/10/2023)
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
    # Will need to perform random seeks through transformed data set.
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
            # Else -> viable option
            else:
                # Random read from transformed file. 
                # Using J[i] rather than J[i]+1, since Python is zero-indexed.
                # Amount to read is num_dimensions words
                # TSET will be (1, num_dimensions), don't forget the MATLAB transpose!
                start_offset = J[i]*self.ctx.num_dimensions*self.ctx.word_size
                # TSET = self.ctx.TDS.tf_random_read(start_offset[0], num_words_random_read)
                TSET = self.ctx.TDS.tf_random_read(start_offset, num_words_random_read)

                # Append squared distance between self.q (could also use self.Q[query_idx, :]) and the vector read from disk to ANS
                # delta_new += np.sum(np.square(X_i - r[i]))
                # self.q and TSET should both be (1, num_dimensions)
                T = np.append(self.ANS, np.sum(np.square(self.q - TSET)))

                # Append J[i] (see L223) to V
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

        # second_stage of this query = i
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
                self._run_phase_one_inmem(i)
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