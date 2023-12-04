from pathlib import Path
from typing import Tuple
import numpy as np
import os
import math
import io


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

    def open(self, *f_open_args, **f_open_kwargs):
        return self.FileWrapper(self, *f_open_args, **f_open_kwargs)

    class FileWrapper:
        def __init__(self, dataset: "VectorFile", *f_open_args, **f_open_kwargs):
            self.f_open_args = f_open_args
            self.f_open_kwargs = f_open_kwargs
            self.dataset = dataset
            self.file = None

        def __getitem__(self, block_ind: int):
            assert self.file is not None, "Every action on this object must be wrapped with a context. Or use " \
                                          "__enter__ & __exit__ explicitly "
            assert 0 <= block_ind < self.dataset.num_blocks, f"Block should be between 0 and {self.dataset.num_blocks}"
            start_index = self.dataset.num_vectors_per_block * block_ind * self.dataset.num_words_per_vector
            num_vectors = self.dataset.num_vectors_per_block if block_ind < (self.dataset.num_blocks - 1) else \
            self.dataset.shape[0] - block_ind * self.dataset.num_vectors_per_block
            count = num_vectors * (self.dataset.shape[1] + self.dataset.offsets[0] + self.dataset.offsets[1])
            self.file.seek(start_index, os.SEEK_SET)
            if self.dataset.big_endian:
                block = np.fromfile(file=self.file, count=count, dtype=self.dataset.stored_dtype).byteswap(
                    inplace=True)
            else:
                block = np.fromfile(file=self.file, count=count, dtype=self.dataset.stored_dtype)

            block = np.reshape(block,
                               (num_vectors, self.dataset.shape[1] + self.dataset.offsets[0] + self.dataset.offsets[1]),
                               order="C")
            block = block[:,
                    self.dataset.offsets[0] if self.dataset.offsets[0] > 0 else None: -self.dataset.offsets[1] if
                    self.dataset.offsets[1] > 0 else None]
            if self.dataset.stored_dtype is not self.dataset.dtype:
                return block.astype(self.dataset.dtype)
            return block

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
        for res in m:
            print(res.shape)
    print("")
