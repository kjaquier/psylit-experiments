import pathlib
import logging
import time
import sys

from features.gen_cascades import BookData
from utils.misc import benchmark, Timer


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, 
                    format='[%(asctime)s %(levelname)s %(name)s (%(funcName)s)] %(message)s',
                    datefmt='%H:%M:%S') #  


def main(book_name: "Book name",
         input_dir: "Folder containing book data",
         output_dir: "Folder in which to generate output",
         min_entities_occ: ("Minimum number of occurrences of entities to extract", 'option', 'n')=100,
         bench_mode: ("Measure execution time", 'flag', 'b')=False):

    if bench_mode:
        t = Timer()
        t.start()

    book = BookData(book_name, input_dir)
    cascades = benchmark(book.get_all_cascades)(min_entities_occurrences=int(min_entities_occ))
    out_filename = pathlib.PurePath(output_dir) / f"{book_name}.csv"
    logging.info('Writing to %s', out_filename)
    cascades.to_csv(out_filename)

    if bench_mode:
        t.stop()
        logging.debug("Execution time: %s", t)

if __name__ == '__main__':
    import plac
    plac.call(main)
