import logging
import sys
import json
import datetime as dt
import pathlib
from glob import glob

import plac

from data import preprocess
from models import nlp_pipeline
from utils.misc import Timer

from parameters import LOGGING_PARAMETERS, PREPARE_PARAMETERS

logging.basicConfig(**LOGGING_PARAMETERS)


def main(input_filename: "Text document to read (UTF-8) - filename or pattern",
         output_dir: "Folder to write output files",
         run_name: ("Name of the run for output files", 'option', 'r')='',
         save_doc: ("Save spacy Doc object", 'flag', 'd')=False,
         save_features: ("Save tokens and features as csv", 'flag', 'f')=False,
         no_save_entities: ("Don't save entities as csv", 'flag', 'e')=False,
         save_meta: ("Write metadata file about the run", 'flag', 'm')=False,
         start: ("Position to read from", 'option', 't0')=0,
         end: ("Position to stop reading after", 'option', 't1')=0,
         benchmark: ("Measure execution time", 'flag', 'x')=False):

    now = dt.datetime.now().isoformat()

    output_path = pathlib.Path(output_dir)
    batch_size = PREPARE_PARAMETERS['batch_size']
    txt_slice = slice(int(start), int(end) or None)    
    
    if benchmark:
        t_tot = Timer()
        t_tot.start()

    files = list(glob(input_filename))
    n_files = len(files)
    for i, filename in enumerate(files):
        if benchmark:
            t_init = Timer()
            t_init.start()
            
        path = pathlib.Path(filename)
        logging.debug("Reading '%s'[%s:%s], with batch size %s", path, txt_slice.start, txt_slice.stop, batch_size)
        txt = preprocess.read_pg(path)
        
        txt = txt[txt_slice]

        nlp = nlp_pipeline.make_nlp(coref_kwargs=PREPARE_PARAMETERS['coref'])
        pipeline = nlp_pipeline.BookParsePipeline(nlp,
                                                  batch_size=batch_size,
                                                  save_entities=(
                                                      not no_save_entities),
                                                  save_data=True,
                                                  save_doc=save_doc,
                                                  save_features=save_features)

        if benchmark:
            t_init.stop()
            logging.info("Read and pipeline init time: %s", t_init)
            t_process = Timer()
            t_process.start()

        if batch_size:
            pipeline.parse_batches(txt)
        else:
            pipeline.parse(txt)

        if benchmark:
            t_process.stop()
            logging.info("Process time: %s", t_process)

        run_name = run_name or path.stem
        suffix = f"__{i}" if n_files > 1 else ""
        
        pipeline.save(output_path, f"{run_name}{suffix}")
            

        if save_meta:
            meta_file = output_path / f"{run_name}.meta.json"
            metadata = {
                'cmd': sys.argv,
                'time': now,
                'input_filename': input_filename,
                'time_init': str(t_init) if benchmark else None,
                'time_process': str(t_process) if benchmark else None,
                'run_name': run_name,
            }

            logging.info(f"Saving meta data to {meta_file}")
            with open(meta_file, 'w') as f:
                json.dump(metadata, f)

    if benchmark:
        t_tot.stop()
        logging.info("Total execution time: %s", t_tot)
        t_tot.reset()
        t_tot.start()

if __name__ == '__main__':
    plac.call(main)
