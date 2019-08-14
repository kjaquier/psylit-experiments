import pathlib
import logging

from features.gen_cascades import BookData
from utils.misc import benchmark, Timer

from parameters import LOGGING_PARAMETERS, PROCESS_PARAMETERS

logging.basicConfig(**LOGGING_PARAMETERS)


def main(input_dir: "Folder containing book data",
         output_dir: "Folder in which to generate output",
         book_name: ("Name of the book for output files", 'option', 'r')='*',
         bench_mode: ("Measure execution time", 'flag', 'x')=False):

    if bench_mode:
        t_tot = Timer()
        t_tot.start()

    input_dir = pathlib.Path(input_dir)
    book_names = {p.with_suffix('').stem for p in input_dir.glob(f"{book_name}.*")}

    logging.info("Found %d book(s)", len(book_names))

    for current_book_name in book_names:
        if bench_mode:
            t = Timer()
            t.start()
        
        logging.info("Processing: '%s'", current_book_name)

        filename_no_ext = input_dir / current_book_name

        book = BookData(data_file=filename_no_ext.with_suffix('.data.csv'),
                        ent_file=filename_no_ext.with_suffix('.ent.csv'),
                        meta_file=filename_no_ext.with_suffix('.meta.json'))

        book2cascades = benchmark(book.get_all_cascades) if bench_mode else book.get_all_cascades
        cascades = book2cascades(min_entities_occurrences=PROCESS_PARAMETERS['min_entities_occurrences'])
        
        out_filename = pathlib.Path(output_dir) / f"{book_name}.csv"
        logging.info('Writing to %s', out_filename)
        out_filename.unlink()
        cascades.to_csv(out_filename)
        
        if bench_mode:
            t.stop()
            logging.info("Process time: %s", t)

    if bench_mode:
        t_tot.stop()
        logging.info("Total execution time: %s", t_tot)

if __name__ == '__main__':
    import plac
    plac.call(main)
