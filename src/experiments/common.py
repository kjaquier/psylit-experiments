from dataclasses import dataclass
import logging
import pathlib
import datetime as dt
from typing import Dict, Tuple, Set, Optional

import pandas as pd

from models.cascades import Cascades
from utils.misc import path_remove_if_exists
from utils.io import file_parts
from parameters import EXPERIMENTS_PARAMETERS

@dataclass
class Setup:
    data_source: Dict[str, pathlib.Path]
    output_dest: pathlib.Path


ResultType = Dict[str, pd.DataFrame]
DataType = Dict[str, pd.DataFrame]


class BaseExperiment:

    exp_name = 'experiment'
    result_keys: Set[str] = set()
    setup_class = Setup

    def __init__(self, setup: Setup, run_name:str, logger=logging.getLogger(), no_rerun=False):
        self.setup = setup
        self.run_name = run_name
        self.results: Optional[ResultType] = None
        self._logger = logger
        self.no_rerun = no_rerun

    def run(self, **kwargs):
        if self.no_rerun and self._has_already_run(self.setup.output_path, self.run_name):
            self._logger.info("Skipped: %s (already exists)", self.setup.output_dest)
            return
        
        data: DataType = self._load_data()
        results = self._execute(data)

        obtained_result_keys = set(results.keys())
        assert self.result_keys == obtained_result_keys, f"Missing result keys: {self.result_keys - obtained_result_keys!r}"

        self.results = results

    def _load_data(self):
        raise NotImplementedError()
    
    def _execute(self, data):
        raise NotImplementedError() 

    def __call__(self):
        # TODO chain experiments
        raise NotImplementedError()

    @classmethod
    def _get_path_for_result(cls, output_path, run_name, res_name, res_type):
        res_format = cls._get_format_for_type(res_type)
        run_output_dir = cls._get_run_output_dir(output_path, run_name)
        return run_output_dir / f"{res_name}{res_format}"

    @classmethod
    def _has_already_run(cls, output_path, run_name):
        return cls._get_run_output_dir(output_path, run_name).exists()

    @classmethod
    def _get_run_output_dir(cls, output_path, run_name):
        return output_path / cls.exp_name / run_name

    @staticmethod
    def _get_format_for_type(data_type):
        if issubclass(data_type, pd.DataFrame):
            return EXPERIMENTS_PARAMETERS['extensions']['dataframe']
        else:
            raise KeyError(f'No format implemented for type {data_type}')

    def save_results(self):
        if not self.results:
            if self.no_rerun:
                return
            raise Exception('Must call run() first!')
        run_output_dir = self._get_run_output_dir(self.setup.output_dest, self.run_name)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        for res_name, res_value in self.results.items():
            output_filename = self._get_path_for_result(self.setup.output_dest, self.run_name, res_name, type(res_value))
            logging.info("Writing to %s", output_filename)
            path_remove_if_exists(output_filename)
            res_value.to_csv(output_filename)  # TODO separate class for reading/writing data in different formats

    @classmethod
    def load_results(cls, output_path, run_name):
        if not cls._has_already_run(output_path, run_name):
            raise Exception(f"No results found in '{output_path}' for run '{run_name}'")
        results = {}
        for res_name in cls.result_keys:
            filename = cls._get_path_for_result(output_path, run_name, res_name, pd.DataFrame)
            logging.info("Loading %s", filename)
            results[res_name] = pd.read_csv(filename)
        return results


class BaseCascadeExperiment(BaseExperiment):

    exp_name = 'cascade_experiment'

    def _load_data(self):
        data_source = self.setup.data_source
        doc_name = file_parts(data_source['doc_path'])[0]
        raw_casc = Cascades.from_csv(data_source['doc_path'])
        return {'cascades': raw_casc}
