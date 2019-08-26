from functools import wraps
from collections import Mapping
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

    def __init__(self, initial_value=0):
        self.t0 = initial_value
        self.elapsed = initial_value

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

    def __format__(self, format_spec):
        if format_spec:
            return time.strftime(format_spec, time.gmtime(self.elapsed))
        else:
            return str(self)

    def __mul__(self, other):
        if isinstance(other, Timer):
            other = other.elapsed
        return Timer(self.elapsed * other)

    def __truediv__(self, other):
        if isinstance(other, Timer):
            other = other.elapsed
        return Timer(self.elapsed / other)

    def __add__(self, other):
        if isinstance(other, Timer):
            other = other.elapsed
        return Timer(self.elapsed + other)

    def __sub__(self, other):
        if isinstance(other, Timer):
            other = other.elapsed
        return Timer(self.elapsed - other)

    def __neg__(self):
        return Timer(-self.elapsed)


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

DEFAULT_PROGRESS_MSG = lambda i, x, n, t_remaining: f"{i} / {n} [estimated time remaining: {t_remaining:%H:%M:%S}]"

def progress(seq, fmt=DEFAULT_PROGRESS_MSG, n=None, print_func=print):
    n = n or len(seq)
    t_total = Timer()
    t = Timer()
    tasks = enumerate(seq)
    for i, x in tasks:
        t_remaining = (t_total / i) * (n-i) if i > 0 else -t_total
        print_func(fmt(i+1, x, n, t_remaining))
        t.reset()
        t.start()
        yield x
        t.stop()
        t_total += t
    
    print_func(f"Done [total elapsed: {t_total:%H:%M:%S}]")



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


def dynamic_dict(mapping_func):
    """Disguise a callable as a dict-like.
    /!\\ doesn't implement __iter__ and __len__"""

    class WrapperMapping:  # pylint: disable=too-few-public-methods

        def __getitem__(self, key):
            return mapping_func(key)

    return WrapperMapping()
