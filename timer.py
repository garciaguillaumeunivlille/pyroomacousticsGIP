import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def getElapsedTime(self):
        """returns elapsed time"""
        return time.perf_counter() - self._start_time
        
    def stop(self, fromText="STOP"):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use  .start() to start it")

        print(f"{fromText} Total elapsed time : {self.getElapsedTime():0.4f} seconds")
        self._start_time = None

    def show(self, fromText="SHOW"):
        """print time elapsed without stopping timer"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        print(f"{fromText} elapsed time : {self.getElapsedTime():0.4f} seconds")


