from functools import wraps
import logging
import time
from math import ceil


def benchmark(f, log_level=logging.DEBUG):
    rename = True
    try:
        name = f.__name__
    except AttributeError:
        try:
            name = f.__class__.__name__
        except AttributeError:
            name = repr(f)
            rename = False

    logger = logging.getLogger(name)

    t = Timer()
    def bench_f(*args, **kwargs):
        t.reset()
        t.start()
        res = f(*args, **kwargs)
        t.stop()
        logger.log(log_level, "Time:  %s", t)
        return res

    if rename:
        bench_f.__name__ = name

    return bench_f


class Timer:

    def __init__(self):
        self.t0 = 0
        self.elapsed = 0
        self.reset()

    def start(self):
        self.t0 = time.perf_counter()

    def pause(self):
        t = time.perf_counter()
        self.elapsed += t - self.t0
        self.t0 = t
        return t

    def stop(self):
        self.pause()
    
    def reset(self):
        self.t0 = 0
        self.elapsed = 0

    def __str__(self):
        return f"{self.elapsed*1000.0:.6n} ms"


class BatchSequence:

    def __init__(self, seq, batch_size):
        self.seq = seq
        self.batch_size = batch_size
        self.__n = len(seq)
        self.__n_batches = int(ceil(self.__n / batch_size))

    def __len__(self):
        return self.__n_batches

    def __iter__(self):
        w = self.batch_size
        seq = self.seq
        for i in range(self.__n_batches):
            yield seq[i*w:(i+1)*w]


def path_remove_if_exists(p):
    if p.exists():
        p.unlink()


def into(wrapper_func):
    """Calls a function on the result of the decorated one
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            return wrapper_func(func(*args, **kwargs))

        return wrapper

    return decorator
