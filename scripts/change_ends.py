import sys
sys.path.append('src')
from pathlib import Path
import numpy as np
from numpy import linalg as LA
import os

class DataSet:
    def __init__(self, path, fname, num_vectors=None, num_dimensions=None, num_blocks=None, word_size=4):
        self.path = path
        self.fname = fname
        self.full_fname = None
        self.full_fname_out = None
        self.num_vectors = num_vectors
        self.num_dimensions = num_dimensions
        self.num_blocks = num_blocks
        self.word_size = word_size # 4 bytes default (float32)
        self.num_words_per_block = 0
        self.num_vectors_per_block = 0
        self.file_handle = None

    #----------------------------------------------------------------------------------------------------------------------------------------    
    def _open_file(self):
        self.file_handle = open(self.full_fname_out, mode="wb")
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _close_file(self):
        self.file_handle.close()
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _initialise(self):

        np.set_printoptions(suppress=True)

        total_file_words = (self.num_vectors * (self.num_dimensions+1))
        # assert total_file_words % self.num_blocks == 0
        assert (total_file_words / self.num_blocks) % (self.num_dimensions + 1) == 0, "Inconsistent number of blocks selected."
        self.num_words_per_block = int(total_file_words / self.num_blocks)
        self.num_vectors_per_block = int(self.num_words_per_block / (self.num_dimensions + 1))
        self.full_fname = os.path.join(self.path, '') + self.fname
        self.full_fname_out = os.path.join(self.path, '') + self.fname + "_swapped"
    
    #----------------------------------------------------------------------------------------------------------------------------------------
    def generate_dataset_block(self,start_offset=0):

        # NOTE: Some architectures use big-endian for data types such as floats, but little-endian for integer types.
        #       This seems to be the case for the current files. Must therefore NOT byte swap the vector descriptors!
        
        block_idx = start_offset

        with open(self.full_fname, mode="rb") as f:
            
            while True:
                f.seek(self.num_words_per_block*block_idx*self.word_size, os.SEEK_SET) # Multiply by word_size since seek wants a byte location.
                block_raw = np.fromfile(file=f, count=self.num_words_per_block, dtype=np.float32) # Before byte swap
                block = block_raw.byteswap()

                if block.size > 0:

                    # Put source file descriptors back again!    
                    for i in range(0, self.num_vectors_per_block):
                        pos = i * (self.num_dimensions + 1)
                        # descriptor = block_raw[descriptor_start:(descriptor_start + 1)]
                        # block[descriptor_start:descriptor_start + self.word_size] = descriptor
                        # print("Restoring descriptor : Before -> ", float.hex(float(block[pos])))
                        block[pos] = block_raw[pos]
                        # print("Restoring descriptor : After  -> ", float.hex(float(block[pos])))
                    yield block
                    block_idx +=1
                else:
                    break
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _build_swapped_file(self):
        
        # print("In _swap_bytes!")

        # Open tp handle write
        self._open_file()

        # Init generator
        gene_ds = self.generate_dataset_block()

        block_count = 0
        write_count = 0        
        # Generator block loop
        for block in gene_ds:

            self.file_handle.write(block)    
            block_count += 1
            write_count += 1    
            
        print()
        print("Blocks Read    : ", str(block_count))
        print("Blocks Written : ", str(write_count))
        print()

        # Close file
        self._close_file()
        
    #----------------------------------------------------------------------------------------------------------------------------------------
    def process(self):
        
        # Initialisations
        self._initialise()

        self._build_swapped_file()

    #----------------------------------------------------------------------------------------------------------------------------------------

def main():

    path = Path('datasets/histo64i64_12103/')    
    fname = 'histo64i64_12103'
    num_vectors = 12103
    num_dimensions = 64
    num_blocks = 7

    dataset = DataSet(path, fname, num_vectors, num_dimensions, num_blocks)
    dataset.process()


if __name__ == "__main__":
    main()