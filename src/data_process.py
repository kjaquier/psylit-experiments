import os
import logging
import time
import sys

from features.gen_cascades import BookData
from utils.misc import benchmark as bench


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, 
                    format='[%(asctime)s %(levelname)s %(name)s (%(funcName)s)] %(message)s',
                    datefmt='%H:%M:%S') #  


def main(book_name: "Book name",
         input_dir: "Folder containing book data",
         output_dir: "Folder in which to generate output",
         min_entities_occ: ("Minimum number of occurrences of entities to extract", 'option', 'n')=100,
         benchmark: ("Measure execution time", 'flag', 'b')=False):

    if benchmark:
        t0 = time.clock()

    book = BookData(book_name, input_dir)
    cascades = bench(book.get_all_cascades)(min_entities_occurrences=int(min_entities_occ))
    out_filename = os.path.join(output_dir, book_name) + '.csv'
    logging.info('Writing to %s', out_filename)
    cascades.to_csv(out_filename)

    if benchmark:
        t = time.clock() - t0
        logging.debug(f"Execution time: {t*1000:.5n}ms")

if __name__ == '__main__':
    import plac
    plac.call(main)
