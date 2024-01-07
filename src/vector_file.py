from pathlib import Path
from typing import Tuple
import numpy as np
import os
import math
import logging

LOGGER = logging.getLogger("project.files")


class VectorFile:
    """ Handles the processing of a single file that contains a vector """

    def __init__(self, path: Path, shape: Tuple[int, int], dtype=np.float32, stored_dtype=None, num_blocks: int = 4,
                 big_endian=False, offsets: Tuple[int, int] = (0, 0)):
        if not stored_dtype:
            stored_dtype = dtype
        self.path = path
        self.shape = shape
        self.offsets = offsets
        self.dtype = dtype
        self.stored_dtype = stored_dtype
        self.big_endian = big_endian
        self.num_blocks = num_blocks
        self.word_size = self.stored_dtype().itemsize
        self.num_vectors_per_block = math.floor(self.shape[0] / self.num_blocks)
        self.num_words_per_vector = (self.shape[1] + self.offsets[0] + self.offsets[1]) * self.word_size
        if self.shape[0] % self.num_blocks > 0:
            LOGGER.warning(f"For file {self.path} the number of vectors({self.shape[0]}) is not divisible by the "
                           f"block number ({self.num_blocks}). This means that the final block can have different "
                           f"shape than the remaining ones")

    def __str__(self):
        return f"({self.shape[0], self.shape[1]} {self.stored_dtype}/{self.dtype} in {self.path})"

    def open(self, *f_open_args, **f_open_kwargs):
        return self.FileWrapper(self, *f_open_args, **f_open_kwargs)

    class FileWrapper:
        def __init__(self, dataset: "VectorFile", *f_open_args, **f_open_kwargs):
            self.f_open_args = f_open_args
            self.f_open_kwargs = f_open_kwargs
            self.dataset = dataset
            self.file = None

        def __assertion_file_check(self):
            assert self.file is not None, "Every action on this object must be wrapped with a context. Or use " \
                                          "__enter__ & __exit__ explicitly "

        def __getitem__(self, block_ind: int):
            assert 0 <= block_ind < self.dataset.num_blocks, f"Block should be between 0 and {self.dataset.num_blocks}"
            start_index = self.dataset.num_vectors_per_block * block_ind * self.dataset.num_words_per_vector
            num_vectors = self.dataset.num_vectors_per_block if block_ind < (self.dataset.num_blocks - 1) else \
                self.dataset.shape[0] - block_ind * self.dataset.num_vectors_per_block
            count = num_vectors * (self.dataset.shape[1] + self.dataset.offsets[0] + self.dataset.offsets[1])
            self.file.seek(start_index, os.SEEK_SET)
            block = np.fromfile(file=self.file, count=count, dtype=self.dataset.stored_dtype)
            if self.dataset.big_endian:
                block = block.byteswap(inplace=True)
            block = np.reshape(block,
                               (num_vectors, self.dataset.shape[1] + self.dataset.offsets[0] + self.dataset.offsets[1]),
                               order="C")
            block = block[:,
                    self.dataset.offsets[0] if self.dataset.offsets[0] > 0 else None: -self.dataset.offsets[1] if
                    self.dataset.offsets[1] > 0 else None]
            if self.dataset.stored_dtype is not self.dataset.dtype:
                return block.astype(self.dataset.dtype)
            return block

        def read_one(self, vector_ind: int) -> np.ndarray:
            assert 0 <= vector_ind < self.dataset.shape[
                0], f"Vector index should be in range (0, {self.dataset.shape[0]}) "
            start_index = self.dataset.num_words_per_vector * vector_ind
            count = (self.dataset.shape[1] + self.dataset.offsets[0] + self.dataset.offsets[1])
            self.file.seek(start_index, os.SEEK_SET)
            vector = np.fromfile(file=self.file, count=count, dtype=self.dataset.stored_dtype)
            if self.dataset.big_endian:
                vector = vector.byteswap(inplace=True)
            vector = np.reshape(vector,
                                (count),
                                order="C")
            vector = vector[
                     self.dataset.offsets[0] if self.dataset.offsets[0] > 0 else None: -self.dataset.offsets[1] if
                     self.dataset.offsets[1] > 0 else None]
            if self.dataset.stored_dtype is not self.dataset.dtype:
                return vector.astype(self.dataset.dtype)
            return vector

        def unsafe_read_all(self) -> np.ndarray:
            self.file.seek(0, os.SEEK_SET)
            block = np.fromfile(file=self.file, dtype=self.dataset.stored_dtype)
            if self.dataset.big_endian:
                block = block.byteswap(inplace=True)
            block = np.reshape(block,
                               (self.dataset.shape[0],
                                self.dataset.shape[1] + self.dataset.offsets[0] + self.dataset.offsets[1]),
                               order="C")
            block = block[:,
                    self.dataset.offsets[0] if self.dataset.offsets[0] > 0 else None: -self.dataset.offsets[1] if
                    self.dataset.offsets[1] > 0 else None]
            if self.dataset.stored_dtype is not self.dataset.dtype:
                return block.astype(self.dataset.dtype)
            return block

        def write(self, val: np.ndarray):
            """ Convert to stored dtype and store """
            self.__assertion_file_check()
            if val.dtype != self.dataset.stored_dtype:
                val = val.astype(self.dataset.stored_dtype)
            self.file.write(val)

        def __getattr__(self, item):
            """ All the methods other that the ones defined here are actually delegated to the file object. Hence
            this can be used as a file in itself as well """
            self.__assertion_file_check()
            return getattr(self.file, item)

        def __iter__(self):
            return self.FileWrapperIterator(self)

        def __enter__(self):
            self.file = self.dataset.path.open(*self.f_open_args, **self.f_open_kwargs)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.file.close()

        class FileWrapperIterator:
            def __init__(self, wrapper: "VectorFile.FileWrapper"):
                self.wrapper = wrapper
                self.block_index = 0

            def __next__(self):
                if self.block_index == self.wrapper.dataset.num_blocks:
                    raise StopIteration
                res = self.wrapper[self.block_index]
                self.block_index += 1
                return res


if __name__ == "__main__":
    a = VectorFile(Path("../datasets/siftsmall/siftsmall"), (10000, 128), num_blocks=5, offsets=(1, 0))
    with a.open() as m:
        all = m.unsafe_read_all()
        i = 0
        for block in m:
            i += block.shape[0]
