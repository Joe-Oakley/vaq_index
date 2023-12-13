from qsession import QSession
from typing import TypedDict, Tuple


class TransformationSummary(TypedDict, total=False):
    """ State that is passed between pieplined transformation elements """
    created: Tuple[str, ...]
    modified: Tuple[str, ...]
    removed: Tuple[str, ...]
    extra: any


class PipelineElement:
    def __init__(self, session: QSession):
        self.session = session

    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        raise NotImplementedError("Child classes should implement this method")
