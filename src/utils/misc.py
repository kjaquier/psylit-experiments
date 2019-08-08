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

    def bench_f(*args, **kwargs):
        t = time.clock()
        res = f(*args, **kwargs)
        t = time.clock() - t
        logger.log(log_level, f"Time:  {t*1000:.3f}ms")
        return res

    if rename:
        bench_f.__name__ = name

    return bench_f

def batch(seq, batch_size, log_level=logging.DEBUG):
    logger = logging.getLogger()
    n = len(seq)
    n_batches = int(ceil(n / batch_size))
    for i in range(n_batches):
        if log_level:
            logger.debug("batch of %d: [%d:%d] / %d", batch_size, i*batch_size, (i+1)*batch_size, n)
        yield seq[i*batch_size:(i+1)*batch_size]
