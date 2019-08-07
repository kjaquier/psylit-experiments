import logging
import time
import sys
import json
import os
import datetime as dt

import plac

from data import preprocess
from models import nlp_pipeline


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, 
                    format='[%(asctime)s %(name)s] %(message)s',
                    datefmt='%H:%M:%S') #  %(levelname)s


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

    pipeline = nlp_pipeline.BookParsePipeline()
    nlp = pipeline.nlp
    logging.debug(f"Pipeline: " + ', '.join(pname for pname, _ in nlp.pipeline))

    if benchmark:
        t_init = time.clock() - t0
        logging.info(f"Read and pipeline init time: {t_init*1000:.5n}ms")
        t1 = time.clock()

    pipeline.parse(txt)

    entities_df = pipeline.get_entities_df()
    df = pipeline.get_df()

    if benchmark:
        t_process = time.clock() - t1
        logging.info(f"Process time: {t_process*1000:.5n}ms")
        t2 = time.clock()

    ent_file = os.path.join(output_dir, run_name) + '.ent.csv'
    doc_file = os.path.join(output_dir, run_name) + '.doc.pkl'
    tok_file = os.path.join(output_dir, run_name) + '.tok.csv'
    data_file = os.path.join(output_dir, run_name) + '.data.csv'
    meta_file = os.path.join(output_dir, run_name) + '.meta.json'

    if not no_save_entities:
        logging.info(f"Saving entities to {ent_file}")
        entities_df.to_csv(ent_file)

    if save_features:
        logging.info(f"Saving features to {tok_file}")
        feat_df = pipeline.get_features_df()
        feat_df.to_csv(tok_file)

    if save_doc:
        logging.info(f"Saving doc object to {doc_file}")
        pipeline.doc.to_disk(save_doc)

    logging.info(f"Saving data to {data_file}")
    df.to_csv(data_file)

    if benchmark:
        t_write = time.clock() - t2
        logging.info(f"Write time: {t_write*1000:.5n}ms")

    if save_meta:
        metadata = {
            'cmd': sys.argv,
            'time': now,
            'input_filename': input_filename,
            'time_init': t_init if benchmark else None,
            'time_process': t_process if benchmark else None,
            'time_write': t_write if benchmark else None,
            'n_predicates': len(df.index),
            'n_corefs': len(entities_df.index),
            'ent_file': ent_file,
            'doc_file': doc_file,
            'data_file': data_file,
        }

        for k in ['n_predicates', 'n_corefs']:
            logging.info(f"{k}: {metadata[k]}")

            logging.info(f"Saving meta data to {meta_file}")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)


if __name__ == '__main__':
    plac.call(main)
