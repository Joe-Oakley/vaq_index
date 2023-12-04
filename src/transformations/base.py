from ..qsession import QSession


class Transformation:
    def __init__(self, session: QSession):
        self.session = session

    def process(self, pipe_state = None):
        raise NotImplementedError("Child classes should implement this method")
