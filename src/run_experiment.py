import pathlib
import logging
from glob import glob

import pyinform

from utils.misc import path_remove_if_exists, progress
from utils.io import file_parts
from models.cascades import Cascades, FEATURE_TRANSFORMERS
import experiments
from parameters import LOGGING_PARAMETERS, ANALYSIS_PARAMETERS

logging.basicConfig(**LOGGING_PARAMETERS)

EXPERIMENTS = {
    c.__name__: c
    for c in [
        experiments.block_entropy.BlockEntropy_StimulusResponse,
    ]
}


def main(experiment: f"Name of experiment to run. Available: {'|'.join(EXPERIMENTS.keys())}",
         input_filename: "File name or pattern of input file",
         output_dir: "Folder to write the results to",
         **arguments):

    Experiment = EXPERIMENTS[experiment]

    BE_NAME = ANALYSIS_PARAMETERS[measure]['name']
    BE_K_VALUES = list(ANALYSIS_PARAMETERS[measure]['k_values'])

    output_path = pathlib.Path(output_dir)

    files = list(glob(input_filename))
    for filename in progress(files, print_func=logging.info):
        path = pathlib.Path(filename)
        input_data['doc_path'] = path
        input_data['doc_name'] = file_parts(path)[0]

        setup = experiments.Setup(
            output_dir=output_path,
            logger=logging.getLogger())

        experiment = Experiment(setup)
        experiment.run()

if __name__ == '__main__':
    import plac
    plac.call(main)
