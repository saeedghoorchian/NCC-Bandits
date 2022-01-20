import pprint
import collections
import contextlib
import time


class Timer:
    def __init__(self):
        self.times = collections.defaultdict(float)

    @contextlib.contextmanager
    def record(self, key):
        begin = time.time()
        yield
        self.times[key] += time.time() - begin

    def _get_str_time(self, seconds):
        hours = seconds // (60 * 60)
        seconds %= 60 * 60
        minutes = seconds // 60
        seconds %= 60
        if hours:
            result = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        elif minutes:
            result = f"{int(minutes)}m {seconds:.2f}s"
        else:
            result = f"{seconds:.2f}s"
        return result

    def print_results(self):
        pprint.pprint({key: self._get_str_time(sec) for key, sec in self.times.items()})
