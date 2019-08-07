import logging
import time

def benchmark(f):
    rename = True
    try:
        name = f.__name__
    except AttributeError:
        try:
            name = f.__class__.__name__
        except AttributeError:
            name = repr(f)
            rename = False

    def bench_f(*args, **kwargs):
        t = time.clock()
        res = f(*args, **kwargs)
        t = time.clock() - t
        logging.debug(f"Time for {name}:  {t*1000:.3n}ms")
        return res

    if rename:
        bench_f.__name__ = name

    return bench_f
