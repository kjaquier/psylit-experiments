import logging
import pathlib
import dataclasses as dc
import datetime as dt
from typing import Dict, Tuple

import pandas as pd

from models.cascades import Cascades
from utils.misc import benchmark, Timer, path_remove_if_exists


@dc.dataclass
class Setup:
    data_source: Dict[str, pathlib.Path]
    output_dir: pathlib.Path
    logger: logging.Logger


ResultType = Dict[Tuple[str, str], pd.DataFrame]
DataType = Dict[str, pd.DataFrame]


class BaseExperiment:

    exp_name = 'experiment'

    def __init__(self, setup: Setup):
        self.setup = setup
        self.results: ResultType = None
        self._logger = self.setup.logger

    def run(self, no_rerun=False, **kwargs):
        if no_rerun and self._has_already_run():
            self._logger.info("Skipped: %s (already exists)", self.setup.output_dir)
        data: DataType = self._load_data()
        res = self._execute(data)
    
    def _load_data(self):
        raise NotImplementedError()
    
    def _execute(self, data):
        raise NotImplementedError() 

    def __call__(self):
        # TODO chain experiments
        raise NotImplementedError()

    def _get_filename_for_result(self, res_name, res_format)
        base = ( f"{exp_name}-{res_name}")
        return base.with_suffix(f".{res_format}")

    def _has_already_run(self):
        return self.setup.output_dir.exists()

    def save_results(self):
        output_dir = self.setup.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for (res_name, res_format), res_df in self.results.items():
            output_filename = output_dir / self._get_filename_for_result(res_name, res_format)
            logging.info("Writing to %s", output_filename)
            path_remove_if_exists(output_filename)
            res_df.to_csv(output_filename)


class BaseCascadeExperiment(BaseExperiment):

    exp_name = 'cascade_experiment'

    def _load_data(self):
        data_source = self.setup.data_source
        raw_casc = Cascades.from_csv(data_source['doc_path'])
        return {'cascades': raw_casc}
