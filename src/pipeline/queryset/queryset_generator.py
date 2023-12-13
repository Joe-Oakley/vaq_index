from pipeline import PipelineElement, TransformationSummary
from vector_file import VectorFile
from pathlib import Path
from qsession import QSession
import os
import numpy as np
import random


class QuerySetGenerator(PipelineElement):
    def __init__(self, session: QSession, num_samples: int):
        super(QuerySetGenerator, self).__init__(session)
        self.num_samples = num_samples

    def __generate_queryset_file(self):
        original_file = self.session.state["ORIGINAL_FILE"]
        self.session.state["QUERYSET_FILE"] = queryset_file = VectorFile(
            Path(os.path.join(self.session.dataset_path, f"queryset_{self.session.fname}")),
            (self.num_samples, original_file.shape[1]),
            original_file.dtype, original_file.stored_dtype, 1)
        all_vector_inds = list(np.arange(0, original_file.shape[0], dtype=np.int32))
        sel = random.sample(all_vector_inds, self.num_samples)
        sel.sort()
        i, j = 0, 0
        with queryset_file.open('wb') as qf:
            with original_file.open("r") as f:
                for block in f:
                    for z in range(i, len(sel)):
                        ind = sel[z]
                        if ind >= (block.shape[0] + j):
                            break
                        qf.write(block[ind - j, :])
                    i = z
                    j += block.shape[0]

    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        self.__generate_queryset_file()
        return {"created": ("QUERYSET_FILE",)}
