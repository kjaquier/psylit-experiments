import pathlib
import logging
from glob import glob

import pyinform

from utils.misc import path_remove_if_exists, progress
from utils.io import file_parts
from models.cascades import Cascades, FEATURE_TRANSFORMERS

from parameters import LOGGING_PARAMETERS, ANALYSIS_PARAMETERS

logging.basicConfig(**LOGGING_PARAMETERS)


def main(measure: "Information-theoretic measure to compute",
         input_filename: "File name or pattern of input file",
         output_dir: "Folder to write the results to"):

    BE_NAME = ANALYSIS_PARAMETERS[measure]['name']
    BE_K_VALUES = list(ANALYSIS_PARAMETERS[measure]['k_values'])

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(glob(input_filename))
    for filename in progress(files, print_func=logging.info):
        path = pathlib.Path(filename)
        doc_name = file_parts(path)[0]

        raw_casc = Cascades.from_csv(path)
        
        transform_function = FEATURE_TRANSFORMERS[ANALYSIS_PARAMETERS[measure]['transformer']]
        casc = transform_function(raw_casc)
        
        B = casc.batch_single_measure(
            'Subject',
            pyinform.blockentropy.block_entropy, 
            BE_NAME, 
            ks=BE_K_VALUES, 
            local=ANALYSIS_PARAMETERS[measure]['local'])
        
        output_filename = (output_path / doc_name).with_suffix(ANALYSIS_PARAMETERS[measure]['extension'])
        logging.info("Writing results to %s", output_filename)
        path_remove_if_exists(output_filename)
        B.to_csv(output_filename)


if __name__ == '__main__':
    import plac
    plac.call(main)
