import pathlib
import logging
from glob import glob

from utils.misc import progress
from utils.io import file_parts
from experiments import block_entropy, transfer_entropy, stim_res
from parameters import LOGGING_PARAMETERS, EXPERIMENTS_PARAMETERS

logging.basicConfig(**LOGGING_PARAMETERS)

EXPERIMENTS = {
    c.__name__: c
    for c in [
        block_entropy.BlockEntropy_StimulusResponse,
        transfer_entropy.TransferEntropy_StimulusResponse,
        transfer_entropy.CompleteTransferEntropy_StimulusResponse,
        stim_res.StimulusResponse,
        stim_res.StimulusResponse_NoSemanticRole,
        stim_res.StimulusResponse_RandSubjUnif,
        stim_res.StimulusResponse_RandSubjProb,
    ]
}


def main(experiment: f"Name of experiment to run. Available: {'|'.join(EXPERIMENTS.keys())}",
         input_filename: "File name or pattern of input file",
         output_dir: "Folder to write the results to",
         from_experiment: ("Input is a directory with the results from another experiment", 'flag', 'e')=False,
         skip_if_exists: ("Skip experiment when result already exists", 'flag', 'k')=False,
         **kwargs):

    Experiment = EXPERIMENTS[experiment]

    output_path = pathlib.Path(output_dir)

    files = list(glob(input_filename, recursive=True))
    logging.info("%s files matching '%s'", len(files), input_filename)
    for filename in progress(files, print_func=logging.info):
        path = pathlib.Path(filename)

        if from_experiment:
            # each experiment is a folder, each run is a subfolder
            doc_path = path 
            run_name = path.name
        else:
            # each run is a file, which folder contains it doesn't matter
            doc_path = path
            run_name = file_parts(path)[0]

        setup = Experiment.make_setup(
            data_source={'doc_path': doc_path},
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
