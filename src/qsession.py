import numpy as np
from numpy import linalg as LA
import os
import json
import timeit
from datetime import datetime

class QSession:

    DEBUG = True
    
    def __init__(self, path, fname, mode = "B", create_qhist=True, use_qhist=True, query_k=2, query_fname=None, num_vectors=None, num_dimensions=None, num_blocks=1, word_size=4, big_endian=False, \
                 q_lambda=1, bit_budget=0, non_uniform_bit_alloc=True, design_boundaries=True, dual_phase=True, inmem_vaqdata=False, relative_dist = True):
        
        self.path                   = path
        self.fname                  = fname
        self.mode                   = mode
        self.create_qhist           = create_qhist
        self.use_qhist              = use_qhist
        self.query_k                = query_k
        self.query_fname            = query_fname
        self.num_vectors            = num_vectors
        self.num_dimensions         = num_dimensions
        self.num_blocks             = num_blocks
        self.word_size              = word_size
        self.big_endian             = big_endian
        self.q_lambda               = q_lambda
        self.bit_budget             = bit_budget
        self.non_uniform_bit_alloc  = non_uniform_bit_alloc
        self.design_boundaries      = design_boundaries
        self.dual_phase             = dual_phase
        self.inmem_vaqdata          = inmem_vaqdata
        self.relative_dist          = relative_dist

        np.set_printoptions(linewidth=200)
                
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _initialise(self):
       
        # Add derived parameters used in > one class
        self.total_file_words = self.num_vectors * (self.num_dimensions + 1)
        self.num_words_per_block = int(self.total_file_words / self.num_blocks)
        self.num_vectors_per_block = int(self.num_words_per_block / (self.num_dimensions + 1))
        self.tf_num_words_per_block = 0
        self.tf_num_vectors_per_block = 0
        self.tp_num_words_per_block = 0
        
        # Variable arrays used in > one class
        self.dim_means = np.zeros((1, self.num_dimensions), dtype=np.float32)
        self.cov_matrix = np.zeros((self.num_dimensions, self.num_dimensions), dtype=np.float32)
        self.transform_matrix = np.zeros((self.num_dimensions, self.num_dimensions), dtype=np.float32)   
        self.cells = None
        self.boundary_vals = None     

        # Basic validations
        assert os.path.isdir(self.path), "Path entered : " + self.path + " does not exist!"
        assert (self.total_file_words / self.num_blocks) % (self.num_dimensions + 1) == 0, "Inconsistent number of blocks selected."
        assert self.mode in ('F','B','Q'), "Mode must be one of F, B or Q"
        assert self.query_k > 0, "query_k must be > 0"
        assert self.num_vectors > 0, "num_vectors must be greater than 0"
        assert self.num_dimensions > 0, "num_dimensions must be greater than 0"
        assert self.num_blocks > 0, "num_blocks must be greater than 0"
        assert self.word_size > 0, "word_size must be greater than 0"
        assert self.q_lambda >= 0, "q_lambda must be greater than or equal to 0"
        assert self.bit_budget > 0, "bit_budget must be greater than 0"
        
        # Print floating-point numbers using a fixed point notation
        np.set_printoptions(suppress=True)
        
    #----------------------------------------------------------------------------------------------------------------------------------------        
    def process_timer(self, metric, start_timer):
        # end_timer = timeit.default_timer()
        # duration = end_timer - start_timer
        # update_stats(metric, duration)
        pass
    #----------------------------------------------------------------------------------------------------------------------------------------
    def debug_timer(self, function, reference_time, message, indent=0):
        tabs = ''
        if QSession.DEBUG:
            for i in range(indent + 1):
                tabs += '\t'
            current_time = timeit.default_timer()
            msg = function + ' -> ' + message
            elapsed = tabs + str(current_time - reference_time)
            print("[TIMER] " , msg , "            Elapsed: ", elapsed)  
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    def run(self):
        
        # Initialisations
        self._initialise()
        
        # Doing these here avoids circular dependency issues
        from dataset import DataSet
        from transformed import TransformedDataSet
        from vaqindex import VAQIndex
        from queryset import QuerySet                
        
        # Composition classes
        self.DS:DataSet = None
        self.TDS:TransformedDataSet = None
        self.VAQ:ValueError = None
        self.QS:QuerySet = None

        print()
        print("Session Begins -> Run Mode ", self.mode)
        print("=============================")
              
        QSession_start_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Session Start Time : ", str(current_time))
        print()
        
        # Proceed according to requested run mode
        
        # Mode B : BUILD -> Build and process DataSet, TransformedDataSet and VAQIndex objects
        # Mode F : FULL  -> Build and process DataSet, Transformed DataSet, VAQIndex and QuerySet objects
        if self.mode in ('B','F'):
            DS_start_time = timeit.default_timer()
            print("Instantiating and processing DataSet")
            self.DS = DataSet(ctx=self)
            self.DS.process()
            self.debug_timer('QSession.run', DS_start_time, "DataSet processing elapsed time")
            print()            
            
            TDS_start_time = timeit.default_timer()
            print("Instantiating and building TransformedDataSet")        
            self.TDS = TransformedDataSet(ctx=self)
            self.TDS.build()
            self.debug_timer('QSession.run', TDS_start_time, "TransformedDataSet processing elapsed time")
            print()            

            VAQ_start_time = timeit.default_timer()
            print("Instantiating and building VAQIndex")        
            self.VAQ = VAQIndex(ctx=self)
            self.VAQ.build()
            self.debug_timer('QSession.run', VAQ_start_time, "VAQIndex processing elapsed time")        
            print()

            # QuerySet only required for Mode F
            if self.mode == 'F':
                QS_start_time = timeit.default_timer()
                print("Instantiating and processing QuerySet")        
                self.QS = QuerySet(ctx=self)
                self.QS.process()
                self.debug_timer('QSession.run', QS_start_time, "QuerySet processing elapsed time")
                print()

        # Mode Q : QUERY -> Instantiate (but do not process/build) TransformedDataSet and VAQIndex objects. Create and process QuerySet object.
        if self.mode == 'Q':
            TDS_start_time = timeit.default_timer()
            print("Instantiating TransformedDataSet")        
            self.TDS = TransformedDataSet(ctx=self)
            self.debug_timer('QSession.run', TDS_start_time, "TransformedDataSet processing elapsed time")
            print()                 
            
            VAQ_start_time = timeit.default_timer()
            print("Instantiating VAQIndex")        
            self.VAQ = VAQIndex(ctx=self)
            self.debug_timer('QSession.run', VAQ_start_time, "VAQIndex processing elapsed time")        
            print()

            QS_start_time = timeit.default_timer()
            print("Instantiating and processing QuerySet")        
            self.QS = QuerySet(ctx=self)
            self.QS.process()
            self.debug_timer('QSession.run', QS_start_time, "QuerySet processing elapsed time")
            print()

                
        QSession_end_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Session End Time : ", current_time, " Elapsed : ", str(QSession_end_time - QSession_start_time) )
        print()

    #----------------------------------------------------------------------------------------------------------------------------------------

