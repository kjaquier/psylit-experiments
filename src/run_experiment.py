import pathlib
import logging
from glob import glob

from utils.misc import progress
from utils.io import file_parts
from experiments import block_entropy as blockent_exp, transfer_entropy as tfr_exp
from parameters import LOGGING_PARAMETERS, EXPERIMENTS_PARAMETERS

logging.basicConfig(**LOGGING_PARAMETERS)

EXPERIMENTS = {
    c.__name__: c
    for c in [
        blockent_exp.BlockEntropy_StimulusResponse,
        tfr_exp.TransferEntropy_StimulusResponse,
        tfr_exp.CompleteTransferEntropy_StimulusResponse,
    ]
}


def main(experiment: f"Name of experiment to run. Available: {'|'.join(EXPERIMENTS.keys())}",
         input_filename: "File name or pattern of input file",
         output_dir: "Folder to write the results to",
         skip_if_exists: ("Skip experiment when result already exists", 'flag', 'k')=False,
         **kwargs):

    Experiment = EXPERIMENTS[experiment]

    output_path = pathlib.Path(output_dir)

    files = list(glob(input_filename))
    logging.info("%s files matching '%s'", len(files), input_filename)
    for filename in progress(files, print_func=logging.info):
        path = pathlib.Path(filename)

        run_name = file_parts(path)[0]

        setup = Experiment.make_setup(
            data_source={'doc_path': path},
            output_dest=output_path,
            **EXPERIMENTS_PARAMETERS['experiments'][Experiment.__name__],
            **kwargs,
        )

        experiment = Experiment(setup, run_name, no_rerun=skip_if_exists)
        experiment.run()
        experiment.save_results()

if __name__ == '__main__':
    import plac
    plac.call(main)
