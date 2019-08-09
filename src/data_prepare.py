import logging
import time
import sys
import json
import datetime as dt

import plac

from data import preprocess
from models import nlp_pipeline


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, 
                    format='[%(asctime)s %(levelname)s %(name)s (%(funcName)s)] %(message)s',
                    datefmt='%H:%M:%S') #  


def main(input_filename: "Raw text of book to read (UTF-8)",
         run_name: "Name of the run for output files",
         output_dir: "Folder to write output files",
         save_doc: ("Save spacy Doc object", 'flag', 'd')=False,
         save_features: ("Save tokens and features as csv", 'flag', 'f')=False,
         no_save_entities: ("Don't save entities as csv", 'flag', 'e')=False,
         save_meta: ("Write metadata file about the run", 'flag', 'm')=False,
         start: ("Position to read from", 'option', 't0')=0,
         end: ("Position to stop reading after", 'option', 't1')=0,
         benchmark: ("Measure execution time", 'flag', 'b')=False):

    now = dt.datetime.now().isoformat()

    start = int(start)
    end = int(end)
    logging.debug(f'start={start}, end={end}')


    if benchmark:
        t0 = time.clock()
    txt = preprocess.read_pg(input_filename)
    n = len(txt)
    txt = txt[start:end] if end else txt[start:]
    logging.info(f'Processing {len(txt)}/{n} chars')

    nlp = nlp_pipeline.make_nlp()
    pipeline = nlp_pipeline.BookParsePipeline(nlp,
                                              output_dir=output_dir,
                                              run_name=run_name,
                                              save_entities=(
                                                  not no_save_entities),
                                              save_data=True,
                                              save_doc=save_doc,
                                              save_features=save_features)

    if benchmark:
        t_init = time.clock() - t0
        logging.info(f"Read and pipeline init time: {t_init*1000:.5n}ms")
        t1 = time.clock()

    pipeline.parse(txt)

    if benchmark:
        t_process = time.clock() - t1
        logging.info(f"Process time: {t_process*1000:.5n}ms")
        #t2 = time.clock()

    if save_meta:
        prefix = pipeline.get_output_prefix()
        meta_file = f"{prefix}.meta.json"
        metadata = {
            'cmd': sys.argv,
            'time': now,
            'input_filename': input_filename,
            'time_init': t_init if benchmark else None,
            'time_process': t_process if benchmark else None,
            'run_name': run_name,
            #'n_predicates': len(df.index),
            #'n_corefs': len(entities_df.index),
        }

        #for k in ['n_predicates', 'n_corefs']:
        #    logging.info(f"{k}: {metadata[k]}")

        logging.info(f"Saving meta data to {meta_file}")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)


if __name__ == '__main__':
    plac.call(main)
