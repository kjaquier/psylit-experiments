from glob import glob
from collections import defaultdict
import pathlib
import logging
import json

import pandas as pd

from features.gen_cascades import BookData
from utils.misc import benchmark, Timer, path_remove_if_exists
from utils.io import file_parts

from parameters import LOGGING_PARAMETERS, PROCESS_PARAMETERS

logging.basicConfig(**LOGGING_PARAMETERS)


def main(input_dir: "Folder containing book data",
         output_dir: "Folder in which to generate output",
         book_name: ("Name of the book for output files", 'option', 'r')='*',
         skip_if_exists: ("Skip books where output already exists", 'option', 'k')=False,
         bench_mode: ("Measure execution time", 'flag', 'x')=False):

    timers = defaultdict(Timer)

    if bench_mode:
        timers['tot'].start()

    input_dir = pathlib.Path(input_dir)
    books = set(input_dir.glob(f"{book_name}{PROCESS_PARAMETERS['extensions']['data_input']}"))
    n_books = len(books)
    logging.info("Found %d book(s)", n_books)

    for i, current_book_path in enumerate(books):
        current_book_name = file_parts(current_book_path)[0]

        if bench_mode:
            timers['process'].start()
            
        logging.info("Processing book %d / %d: '%s'", i+1, n_books, current_book_name)


        filename_no_ext = input_dir / current_book_name
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        out_filename = output_path / f"{current_book_name}{PROCESS_PARAMETERS['extensions']['cascades']}"
        if skip_if_exists and out_filename.exists():
            logging.info("Skipped: %s (already exists)", out_filename)
            continue

        # Read data files
        data_df = pd.read_csv(current_book_path, index_col=False)
        ent_input_pattern = filename_no_ext.with_suffix(PROCESS_PARAMETERS['extensions']['entities_input'])
        # logging.info('Looking for %s',ent_input_pattern)
        ent_input_file = next(iter(glob(str(ent_input_pattern))))
        # logging.info('Found entities file: %s',ent_input_file)
        ents_df = pd.read_csv(ent_input_file, index_col=0)
        with open(filename_no_ext.with_suffix(PROCESS_PARAMETERS['extensions']['metadata'])) as f:
            metadata = json.load(f)

        # Generate cascades
        book = BookData(data_df=data_df,
                        ents_df=ents_df,
                        **metadata)
        book2cascades = benchmark(book.get_all_cascades) if bench_mode else book.get_all_cascades
        cascades = book2cascades(min_entities_occurrences=PROCESS_PARAMETERS['min_entities_occurrences'])
        
        
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
