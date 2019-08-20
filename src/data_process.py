from collections import defaultdict
import pathlib
import logging
import json

import pandas as pd

from features.gen_cascades import BookData
from utils.misc import benchmark, Timer, path_remove_if_exists

from parameters import LOGGING_PARAMETERS, PROCESS_PARAMETERS

logging.basicConfig(**LOGGING_PARAMETERS)


def main(input_dir: "Folder containing book data",
         output_dir: "Folder in which to generate output",
         book_name: ("Name of the book for output files", 'option', 'r')='*',
         bench_mode: ("Measure execution time", 'flag', 'x')=False):

    timers = defaultdict(Timer)

    if bench_mode:
        timers['tot'].start()

    input_dir = pathlib.Path(input_dir)
    book_names = {p.with_suffix('').stem for p in input_dir.glob(f"{book_name}.*")}
    n_books = len(book_names)
    logging.info("Found %d book(s)", n_books)

    for i, current_book_name in enumerate(book_names):
        if bench_mode:
            timers['process'].start()
            
        logging.info("Processing book %d / %d: '%s'", i+1, n_books, current_book_name)

        # Read data files
        filename_no_ext = input_dir / current_book_name
        data_df = pd.read_csv(filename_no_ext.with_suffix('.data.csv'), index_col=False)
        ents_df = pd.read_csv(filename_no_ext.with_suffix('.ent.csv'), index_col=0)
        with open(filename_no_ext.with_suffix('.meta.json')) as f:
            metadata = json.load(f)

        # Generate cascades
        book = BookData(data_df=data_df,
                        ents_df=ents_df,
                        **metadata)
        book2cascades = benchmark(book.get_all_cascades) if bench_mode else book.get_all_cascades
        cascades = book2cascades(min_entities_occurrences=PROCESS_PARAMETERS['min_entities_occurrences'])
        
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        out_filename = output_path / f"{current_book_name}.csv"
        logging.info('Writing to %s', out_filename)
        
        if bench_mode:
            timers['write'].start()

        path_remove_if_exists(out_filename)
        cascades.to_csv(out_filename)
        
        if bench_mode:
            timers['write'].stop()
            timers['process'].stop()
            logging.info("Writing time: %s", timers['write'])
            logging.info("Process time: %s", timers['process'])

    if bench_mode:
        timers['tot'].stop()
        logging.info("Total execution time: %s", timers['tot'])

if __name__ == '__main__':
    import plac
    plac.call(main)
