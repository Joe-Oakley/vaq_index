import numpy as np
import os

from qsession import QSession


class TransformedDataSet:
    def __init__(self, ctx: QSession = None):
        self.ctx                = ctx
        self.full_tf_fname      = None
        self.full_tp_fname      = None
        self.tf_handle_read     = None
        self.tf_handle_write    = None
        self.tp_handle_read     = None
        self.tp_handle_write    = None
        
        # Initialisations
        self._initialise()

    # ----------------------------------------------------------------------------------------------------------------------------------------
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
            raise ValueError("Invalid ftype selected: ", ftype)

    # # ----------------------------------------------------------------------------------------------------------------------------------------
    # def _close_file(self, handle):

    #     if handle == self.tf_handle_read:
    #         self.tf_handle_read = None
    #     elif handle == self.tf_handle_write:
    #         self.tf_handle_write = None
    #     elif handle == self.tp_handle_read:
    #         self.tp_handle_read = None
    #     elif handle == self.tp_handle_write:
    #         self.tp_handle_write = None
    #     else:
    #         raise ValueError("Invalid handle given to _close_file().")

    #     handle.close()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self, ftype, mode):

        if ftype == 'tf':
            if mode == 'rb':
                self.tf_handle_read.close()
                self.tf_handle_read = None
            elif mode == 'wb':
                self.tf_handle_write.close()
                self.tf_handle_write = None
            else:
                raise ValueError("Invalid mode selected: ", mode)
        elif ftype == 'tp':
            if mode == 'rb':
                self.tp_handle_write.close()
                self.tp_handle_read = None
            elif mode == 'wb':
                self.tp_handle_write.close()
                self.tp_handle_write = None
            else:
                raise ValueError("Invalid mode selected: ", mode)
        else:
            raise ValueError("Invalid ftype selected: ", ftype)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialise(self):

        np.set_printoptions(suppress=True)
        
        # Load DataSet means, covariance and transform matrices
        self._load_dataset_vars()

        # Initialise variables. Note that "context" variables in the parent class/object are accessed/set, as well as local properties (filenames)
        self.full_tf_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.tf'
        self.full_tp_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.tp'
        
        # Calculate new properties for number of words/vectors per block in transformed dataset. There are fewer words in this dataset,
        # since identifiers have been removed.
        total_tf_file_words = (self.ctx.num_vectors * (self.ctx.num_dimensions))
        print("total_tf_file_words: ", str(total_tf_file_words))

        assert (total_tf_file_words % self.ctx.num_blocks == 0) and (total_tf_file_words // self.ctx.num_blocks) % self.ctx.num_dimensions == 0, "Incorrect number of blocks specified"

        self.ctx.tf_num_words_per_block = int(total_tf_file_words / self.ctx.num_blocks)
        print("self.ctx.tf_num_words_per_block: ", str(self.ctx.tf_num_words_per_block))

        self.ctx.tf_num_vectors_per_block = int(self.ctx.tf_num_words_per_block / (self.ctx.num_dimensions))
        print("self.ctx.tf_num_vectors_per_block: ", str(self.ctx.tf_num_vectors_per_block))

        self.ctx.tp_num_words_per_block = int(total_tf_file_words / self.ctx.num_dimensions)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_dataset_vars(self):

        dataset_full_varfile = os.path.join(self.ctx.path, '') + self._find_file_by_suffix('.dsvars.npz')

        print("Loading dataset variables from ", dataset_full_varfile)
        with np.load(dataset_full_varfile) as data:
            self.ctx.dim_means = data['DIM_MEANS']
            self.ctx.cov_matrix = data['COV_MATRIX']
            self.ctx.transform_matrix = data['TRANSFORM_MATRIX']

    # ----------------------------------------------------------------------------------------------------------------------------------------
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
    def _build_tf(self):

        print("Arrived at _build_tf!")

        # Get file handle for tf write
        self._open_file('tf', 'wb')

        # Calculate repmean; should only be in this function for mode B, so access dim_means from DS property
        rep_mean = np.tile(self.ctx.dim_means, (self.ctx.num_vectors_per_block, 1))
        
        gene = self.ctx.DS.generate_dataset_block()       
        for X in gene:
            Y = np.subtract(X, rep_mean)
            Z = np.matmul(Y, self.ctx.transform_matrix)
            self.tf_handle_write.write(Z)

        self._close_file('tf', 'wb')

        print("Finished _build_tf!")

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def generate_tf_block(self, start_offset=0):

        if self.ctx.mode in ('F','B'):

            block_idx = start_offset

            with open(self.full_tf_fname, mode="rb") as f:
                
                while True:
                    f.seek(self.ctx.tf_num_words_per_block*block_idx*self.ctx.word_size, os.SEEK_SET)
                    block = np.fromfile(file=f, count=self.ctx.tf_num_words_per_block, dtype=np.float32)

                    if block.size > 0:
                        block = np.reshape(block, (self.ctx.tf_num_vectors_per_block, self.ctx.num_dimensions), order="C")
                        yield block
                        block_idx +=1
                    else:
                        break
        else:
            raise ValueError("Entered generate_tf_block outside of modes B or F.")

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _build_tp(self):

        print("Arrived at _build_tp!")
        write_count = 0
        
        # Open tp handle write
        self._open_file('tp', 'wb')

        for i in range(self.ctx.num_dimensions):

            XX = np.zeros(self.ctx.num_vectors, dtype=np.float32)

            # Init generator + loop var
            gene_tf = self.generate_tf_block()
            block_count = 0

            for X in gene_tf:
                
                XX[(block_count * self.ctx.tf_num_vectors_per_block):((block_count + 1) * self.ctx.tf_num_vectors_per_block)] = X[:,i]  
                block_count += 1

            # Write current dimension to tp handle write
            self.tp_handle_write.write(XX)

            write_count += 1

        # Close tp handle write
        self._close_file('tp', 'wb')

        print("Finished _build_tp!") 

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Gives (num_vectors, 1) block of tp; all data for a single dimension.
    def generate_tp_block(self, start_offset=0):

        if self.ctx.mode in ('F','B'):

            block_idx = start_offset

            with open(self.full_tp_fname, mode="rb") as f:

                while True:
                    f.seek(self.ctx.tp_num_words_per_block * block_idx * self.ctx.word_size, os.SEEK_SET)
                    block = np.fromfile(file=f, count=self.ctx.tp_num_words_per_block, dtype=np.float32)

                    if block.size > 0:
                        block = np.reshape(block, (self.ctx.num_vectors, 1), order="C")  # Order F to mirror MATLAB
                        yield block
                        block_idx += 1
                    else:
                        break
        else:
            raise ValueError("Entered generate_tp_block outside of modes F and B.")

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def tf_random_read(self, start_offset,
                       num_words_random_read):  # num_words_random_read is like count in other read functions.

        self._open_file('tf', 'rb')
        self.tf_handle_read.seek(start_offset, os.SEEK_SET)
        block = np.fromfile(file=self.tf_handle_read, count=num_words_random_read, dtype=np.float32)

        if block.size > 0:
            block = np.reshape(block,
                               (1, self.ctx.num_dimensions))  # Done this way round, rather than MATLAB [DIMENSION, 1]'.

        return block

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def build(self):

        # Build tf - Initial dataset transformed by KLT matrix (generated in dataset.py)
        self._build_tf()

        # Build tp - Transposed version of transformed dataset. New format: All values for dim 1, all values for dim 2, etc...
        self._build_tp()

    # ----------------------------------------------------------------------------------------------------------------------------------------
