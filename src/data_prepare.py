from collections import defaultdict
import logging
import sys
import datetime as dt
import pathlib
from glob import glob

import plac

from data import preprocess
from models import nlp_pipeline
from utils.misc import Timer, BatchSequence

from parameters import LOGGING_PARAMETERS, PREPARE_PARAMETERS

logging.basicConfig(**LOGGING_PARAMETERS)


def main(input_filename: "Text document to read (UTF-8) - filename or pattern",
         output_dir: "Folder to write output files",
         run_name: ("Name of the run for output files", 'option', 'r')='',
         save_doc: ("Save spacy Doc object", 'flag', 'd')=False,
         save_features: ("Save tokens and features as csv", 'flag', 'f')=False,
         no_save_entities: ("Don't save entities as csv", 'flag', 'e')=False,
         no_save_meta: ("Write metadata file about the run", 'flag', 'm')=False,
         start: ("Position to read from", 'option', 't0')=0,
         end: ("Position to stop reading after", 'option', 't1')=0,
         bench_mode: ("Measure execution time", 'flag', 'x')=False):

    now = dt.datetime.now().isoformat()
    timers = defaultdict(Timer)

    output_path = pathlib.Path(output_dir)
    batch_size = PREPARE_PARAMETERS['batch_size']
    txt_slice = slice(int(start), int(end) or None)    
    
    if bench_mode:
        timers['tot'].start()
        timers['init'].start()

    nlp = nlp_pipeline.make_nlp(coref_kwargs=PREPARE_PARAMETERS['coref'],
                                lexicon_kwargs=PREPARE_PARAMETERS['lexicon'])
    pipeline = nlp_pipeline.BookParsePipeline(nlp,
                                              save_entities=(not no_save_entities),
                                              save_data=True,
                                              save_doc=save_doc,
                                              save_meta=(not no_save_meta),
                                              save_features=save_features)

    if bench_mode:
        timers['init'].stop()
        logging.info("Pipeline init time: %s", timers['init'])

    files = list(glob(input_filename))
    n_files = len(files)
    for i, filename in enumerate(files):
        if bench_mode:
            timers['read'].start()
            
        path = pathlib.Path(filename)
        logging.info("Reading file %d / %d: '%s'[%s:%s:%s]", i+1,
                     n_files, path, txt_slice.start, txt_slice.stop, batch_size)

        txt = preprocess.read_pg(path)
        txt = txt[txt_slice]

        if bench_mode:
            timers['read'].stop()
            timers['process'].start()

        if batch_size:
            texts = BatchSequence(txt, batch_size)
            pipeline.parse_batches(texts)
        else:
            pipeline.parse(txt)

        if bench_mode:
            timers['process'].stop()
            logging.info("Process time: %s", timers['process'])
            timers['write'].start()

        metadata = {
            'cmd': sys.argv,
            'time': now,
            'input_filename': input_filename,
            'run_name': run_name or path.stem,
        }
        # TODO add run parameters
        if bench_mode:
            metadata.update({
                'time_read': str(timers['read']),
                'time_init': str(timers['init']),
                'time_process': str(timers['process']),
            })

        run_name_suffix = f"__{i}" if n_files > 1 else ""
    
        pipeline.save(output_path,
                      run_name=f"{metadata['run_name']}{run_name_suffix}",
                      metadata=metadata)

        if bench_mode:
            timers['write'].stop()
            logging.debug("Write time: %s", timers['write'])

    if bench_mode:
        timers['tot'].stop()
        logging.info("Total execution time: %s", timers['tot'])

if __name__ == '__main__':
    plac.call(main)
