import os
import logging
import time
import sys

from book2cascades import doc2cascade

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def bench(f):
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
        print(f"[Time] {name} {t*1000:.3n}ms")
        return res

    if rename:
        bench_f.__name__ = name

    return bench_f


def main(book_name:"Book name",
         data_dir:"Folder containing book data",
         benchmark:("Measure execution time", 'flag', 'b')=False):

    if benchmark:
        t0 = time.clock()

    book = doc2cascade.BookData(book_name, data_dir)
    cascades = bench(book.get_all_cascades)()
    

    if benchmark:
        t = time.clock() - t0
        print(f"Execution time: {t*1000:.5n}ms")

    for ent, casc in cascades.items():
        print(ent, len(casc.index))

if __name__ == '__main__':
    print(os.getcwd())
    import plac
    plac.call(main)