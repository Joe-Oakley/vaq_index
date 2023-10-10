import numpy as np
from numpy import linalg as LA
import os

from dataset import DataSet

class TransformedDataSet:
    def __init__(self, path, mode='B', DS:DataSet=None):
        self.path = path
        self.full_tf_fname = None
        self.full_tp_fname = None
        self.tf_num_words_per_block = 0
        self.tf_num_vectors_per_block = 0
        self.tp_num_words_per_block = 0

        self.dim_means = None
        self.cov_matrix = None
        self.transform_matrix = None

        self.mode = mode
        self.DS = DS

        self.tf_handle_read = None
        self.tf_handle_write = None
        self.tp_handle_read = None
        self.tp_handle_write = None
    #----------------------------------------------------------------------------------------------------------------------------------------    
    def _open_file(self, ftype, mode):

        if ftype == 'tf':
            if mode == 'rb':
                self.tf_handle_read = open(self.full_tf_fname, mode=mode)
            elif mode == 'wb':
                self.tf_handle_write = open(self.full_tf_fname, mode=mode)
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'tp':
            if mode == 'rb':
                self.tp_handle_read = open(self.full_tp_fname, mode=mode)
            elif mode == 'wb':
                self.tp_handle_write = open(self.full_tp_fname, mode=mode)
            else:
                raise ValueError("Invalid mode selected: ", mode)
        else:
            raise ValueError("Invalid ftype selected: ", mode)

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self, handle):

        if handle == self.tf_handle_read:
            self.tf_handle_read = None
        elif handle == self.tf_handle_write:
            self.tf_handle_write = None
        elif handle == self.tp_handle_read:
            self.tp_handle_read = None
        elif handle == self.tp_handle_write:
            self.tp_handle_write = None
        else:
            raise ValueError("Invalid handle given to _close_file().")

        handle.close()

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _initialise(self):

        np.set_printoptions(suppress=True)

        # We are passing in a dataset object. Not explicitly copying dataset vars from DataSet object; we will 
        # access them from the object when needed.
        if self.mode == 'B':

            # If DS isn't a dataset object, complain
            if not isinstance(self.DS, DataSet):
                raise ValueError("DataSet object not successfully passed in.")
            
            self.full_tf_fname = os.path.join(self.path, '') + self.DS.fname + '.tf'
            self.full_tp_fname = os.path.join(self.path, '') + self.DS.fname + '.tp'
            
            # Calculate new properties for number of words/vectors per block in transformed dataset. There are fewer words in this dataset,
            # since identifiers have been removed.
            total_tf_file_words = (self.DS.num_vectors * (self.DS.num_dimensions))
            print("total_tf_file_words: ", str(total_tf_file_words))

            self.tf_num_words_per_block = int(total_tf_file_words / self.DS.num_blocks)
            print("self.tf_num_words_per_block: ", str(self.tf_num_words_per_block))

            self.tf_num_vectors_per_block = int(self.tf_num_words_per_block / (self.DS.num_dimensions))
            print("self.tf_num_vectors_per_block: ", str(self.tf_num_vectors_per_block))

            self.tp_num_words_per_block = int(total_tf_file_words / self.DS.num_dimensions)
            
        # We aren't passing in a dataset object
        elif self.mode == 'L':
            self._load_dataset_vars()
            self.full_tf_fname = os.path.join(self.path, '') + self._find_file_by_suffix('.tf')
            self.full_tp_fname = os.path.join(self.path, '') + self._find_file_by_suffix('.tp')

            # Todo: update tf words per block etc if needed
        else:
            raise ValueError("Mode must be B (build) or L (load).")

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _load_dataset_vars(self):

        dataset_full_varfile = os.path.join(self.path, '') + self._find_file_by_suffix('.dsvar')

        print("Loading dataset variables from ", dataset_full_varfile)
        with np.load(dataset_full_varfile) as data:
            self.dim_means = data['DIM_MEANS']
            self.cov_matrix = data['COV_MATRIX']
            self.transform_matrix = data['TRANSFORM_MATRIX']

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _find_file_by_suffix(self, suffix):
        hit_count = 0
        hits = []
        for file in os.listdir(self.path):
            if file.endswith(suffix):
                hits.append(file)
                hit_count += 1
        if hit_count > 1:
            raise ValueError("Too many hits for suffix ", str(suffix))
        else:
            return hits[0]
         
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _build_tf(self):

        if self.mode == 'B':

            print("Arrived at _build_tf!")

            # Get file handle for tf write
            self._open_file('tf', 'wb')

            # Calculate repmean; should only be in this function for mode B, so access dim_means from DS property
            rep_mean = np.tile(self.DS.dim_means, (1, self.DS.num_vectors_per_block))
            
            gene = self.DS.generate_dataset_block()       
            for X in gene:
                Y = np.subtract(X, rep_mean)
                Z = np.matmul(self.DS.transform_matrix, Y)
                self.tf_handle_write.write(Z)

            self._close_file(self.tf_handle_write)

            print("Finished _build_tf!")

        else: 
            raise ValueError("Entered _build_tf outside of mode B.")
        
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Gives a (64, 1729) block of transformed data    
    def generate_tf_block(self, start_offset=0):
    

        if self.mode == 'B':

            block_idx = start_offset

            with open(self.full_tf_fname, mode="rb") as f:
                
                while True:
                    f.seek(self.tf_num_words_per_block*block_idx*self.DS.word_size, os.SEEK_SET)
                    block = np.fromfile(file=f, count=self.tf_num_words_per_block, dtype=np.float32)

                    if block.size > 0:
                        block = np.reshape(block, (self.DS.num_dimensions, self.tf_num_vectors_per_block), order="F")
                        yield block
                        block_idx +=1
                    else:
                        break
        else:
            raise ValueError("Entered generate_tf_block outside of mode B.")

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _build_tp(self):

        # Open tp handle write
        self._open_file('tp', 'wb')

        print("Arrived at _build_tp!")
        write_count = 0

        for i in range(self.DS.num_dimensions):

            XX = np.zeros(self.DS.num_vectors, dtype=np.float32)

            # Init generator
            gene_tf = self.generate_tf_block()

            block_count = 0
            # Generator block loop
            for X in gene_tf:

               # Set a row of XX to some function of block 
                XX[(block_count * self.tf_num_vectors_per_block):((block_count + 1) * self.tf_num_vectors_per_block)] = X[i,:] #.reshape(1,self.tf_num_vectors_per_block)
                block_count += 1

            # Write current dimension to tp handle write
            self.tp_handle_write.write(XX)

            write_count += 1

        # Close tp handle write
        self._close_file(self.tp_handle_write)
        
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Gives (num_vectors, 1) block of tp; all data for a single dimension.
    def generate_tp_block(self, start_offset=0):

        if self.mode == 'B':

            block_idx = start_offset

            with open(self.full_tp_fname, mode="rb") as f:
                
                while True:
                    f.seek(self.tp_num_words_per_block*block_idx*self.DS.word_size, os.SEEK_SET)
                    block = np.fromfile(file=f, count=self.tp_num_words_per_block, dtype=np.float32)

                    if block.size > 0:

                        block = np.reshape(block, (self.DS.num_vectors, 1), order="F") # Order F to mirror MATLAB
                        yield block
                        block_idx +=1
                    else:
                        break
        else:
            raise ValueError("Entered generate_tp_block outside of mode B.")

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _load_dataset_vars(self):

        print("Loading dataset variables from ", self.output_path)
        with np.load(self.output_path) as data:
            self.dim_means = data['DIM_MEANS']
            self.cov_matrix = data['COV_MATRIX']
            self.transform_matrix = data['TRANSFORM_MATRIX']

    #----------------------------------------------------------------------------------------------------------------------------------------
    def tf_random_read(self, start_offset, num_words_random_read): # num_words_random_read is like count in other read functions.
         
        self._open_file('tf','rb')
        self.tf_handle_read.seek(start_offset, os.SEEK_SET)
        block = np.fromfile(file=self.tf_handle_read, count = num_words_random_read, dtype=np.float32)

        if block.size > 0:
            block = np.reshape(block, (1, self.DS.num_dimensions)) # Done this way round, rather than MATLAB [DIMENSION, 1]'.

        return block

    #----------------------------------------------------------------------------------------------------------------------------------------
    def build(self):
        
        # Initialisations
        self._initialise()

        # Build tf - Initial dataset transformed by KLT matrix (generated in dataset.py)
        self._build_tf()

        # Build tp - Transposed version of transformed dataset. New format: All values for dim 1, all values for dim 2, etc...
        self._build_tp()

    #----------------------------------------------------------------------------------------------------------------------------------------
