import numpy as np
import os
from pathlib import Path
from typing import Dict, Literal, Tuple, Union, Callable
from vector_file import VectorFile
import timeit
from datetime import datetime


class QSession:
    DEBUG = True

    def __init__(self, path, fname, shape: Tuple[int, int], mode: Literal['F', 'B', 'Q', 'R', 'P'] = "B",
                 create_qhist=True, use_qhist=True,
                 query_k=2, query_fname=None,
                 qhist_fname=None, num_blocks=1, word_size=4, big_endian=False, \
                 q_lambda=1, bit_budget=0, non_uniform_bit_alloc=True, design_boundaries=True, dual_phase=True,
                 inmem_vaqdata=False, relative_dist=True, vecs_to_print=None):

        self.dataset_path = path
        self.fname = fname
        self.query_fname = query_fname
        self.qhist_fname = qhist_fname
        self.mode = mode
        self.create_qhist = create_qhist
        self.use_qhist = use_qhist
        self.query_k = query_k
        self.shape = shape
        self.num_blocks = num_blocks
        self.big_endian = big_endian
        self.q_lambda = q_lambda
        self.bit_budget = bit_budget
        self.non_uniform_bit_alloc = non_uniform_bit_alloc
        self.design_boundaries = design_boundaries
        self.dual_phase = dual_phase
        self.inmem_vaqdata = inmem_vaqdata
        self.relative_dist = relative_dist
        self.vecs_to_print = vecs_to_print

        # ----------- COMPUTED STATE
        self.state: Dict[str, Union[np.ndarray, VectorFile, Callable]] = {
            "ORIGINAL_FILE": VectorFile(Path(os.path.join(self.dataset_path, self.fname)), self.shape,
                                        big_endian=self.big_endian, offsets=(1, 0))
        }
        self.__pipeline = []
        # ------------ OTHER
        np.set_printoptions(linewidth=200)
        np.set_printoptions(suppress=True)
        assert os.path.isdir(self.dataset_path), "Path entered : " + self.dataset_path + " does not exist!"
        if self.mode == 'P' and self.vecs_to_print == None:
            print('Mode P requires a list of Vector IDs to be provided!')
            exit(1)

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def process_timer(self, metric, start_timer):
        # end_timer = timeit.default_timer()
        # duration = end_timer - start_timer
        # update_stats(metric, duration)
        pass

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def debug_timer(self, function, reference_time, message, indent=0):
        tabs = ''
        if QSession.DEBUG:
            for i in range(indent + 1):
                tabs += '\t'
            current_time = timeit.default_timer()
            msg = function + ' -> ' + message
            elapsed = tabs + str(current_time - reference_time)
            print("[TIMER] ", msg, "            Elapsed: ", elapsed)

            # ----------------------------------------------------------------------------------------------------------------------------------------

    def run(self):
        from pipeline.transformations import KLT
        from pipeline.indexes import VAQIndex
        from pipeline.queryset import RandomQuerySetGenerator, VAQQuerySet, RepeatingQuerysetGenerator
        from pipeline.analysis import AnalyseVaqIndex
        self.__pipeline.extend([
            KLT(self),
            RandomQuerySetGenerator(self, 1000),
            VAQIndex(self, self.non_uniform_bit_alloc, self.bit_budget, self.design_boundaries,
                     self.use_qhist),
            VAQQuerySet(self, self.query_k),
            AnalyseVaqIndex(self),
        ])
        self.__run()

    def __run(self):
        prev_result = None
        for element in self.__pipeline:
            prev_result = element.process(prev_result)

    def run_old(self):

        # Doing these here avoids circular dependency issues
        from src.pipeline.dataset import DataSet
        from transformed import TransformedDataSet
        from vaqindex import VAQIndex
        from queryset import QuerySet

        # Composition classes
        self.DS: DataSet = None
        self.TDS: TransformedDataSet = None
        self.VAQ: ValueError = None
        self.QS: QuerySet = None

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
        if self.mode in ('B', 'F'):
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
        elif self.mode == 'Q':
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

        # Mode R : REBUILD -> Instantiate (but do not process/build) TransformedDataSet and VAQIndex objects. Call Rebuild of VAQIndex
        elif self.mode == 'R':
            TDS_start_time = timeit.default_timer()
            print("Instantiating TransformedDataSet")
            self.TDS = TransformedDataSet(ctx=self)
            self.debug_timer('QSession.run', TDS_start_time, "TransformedDataSet processing elapsed time")
            print()

            VAQ_start_time = timeit.default_timer()
            print("Instantiating VAQIndex")
            self.VAQ = VAQIndex(ctx=self)
            self.VAQ.rebuild()
            self.debug_timer('QSession.run', VAQ_start_time, "VAQIndex processing elapsed time")
            print()

        # Mode P : PRINT -> Instantiate (but do not process/build) TransformedDataSet and VAQIndex objects. Call prvd function of VAQIndex
        elif self.mode == 'P':
            TDS_start_time = timeit.default_timer()
            print("Instantiating TransformedDataSet")
            self.TDS = TransformedDataSet(ctx=self)
            self.debug_timer('QSession.run', TDS_start_time, "TransformedDataSet processing elapsed time")
            print()

            VAQ_start_time = timeit.default_timer()
            print("Instantiating VAQIndex")
            self.VAQ = VAQIndex(ctx=self)
            self.VAQ.prvd()
            self.debug_timer('QSession.run', VAQ_start_time, "VAQIndex processing elapsed time")
            print()

        QSession_end_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Session End Time : ", current_time, " Elapsed : ", str(QSession_end_time - QSession_start_time))
        print()

    # ----------------------------------------------------------------------------------------------------------------------------------------
