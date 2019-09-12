from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import logging
import pathlib
from typing import Dict, Set, Optional, Iterable, Tuple

import pandas as pd

from models.cascades import Cascades
from utils.misc import path_remove_if_exists
from parameters import EXPERIMENTS_PARAMETERS

@dataclass
class Setup:
    data_source: Dict[str, pathlib.Path]
    output_dest: pathlib.Path


ResultType = Dict[str, pd.DataFrame]
DataType = Dict[str, pd.DataFrame]


class BaseStore(ABC):
    # should use this to properly decouple Experiment from underlying data sources and formats
    pass


class BaseExperiment(ABC):

    exp_name = 'experiment'
    result_keys: Set[str] = set()
    setup_class = Setup

    def __init__(self, setup: Setup, run_name: str, logger=logging.getLogger(), no_rerun=False):
        self.setup = setup
        self.run_name = run_name
        self.results: Optional[ResultType] = None
        self._logger = logger
        self.no_rerun = no_rerun

    def run(self, **kwargs):
        if self.no_rerun and self._has_already_run(self.setup.output_dest, self.run_name):
            self._logger.info("Skipped: %s (already exists)", self.run_name)
            return
        
        data: DataType = self._load_data()
        results = self._execute(data, **kwargs)

        obtained_result_keys = set(results.keys())
        assert self.result_keys == obtained_result_keys, f"Missing result keys: {self.result_keys - obtained_result_keys!r}"

        self.results = results

    @classmethod
    def make_setup(cls, *args, **kwargs):
        return cls.setup_class(*args, **kwargs)

    def _load_data(self):
        raise NotImplementedError()
    
    def _execute(self, data, **kwargs):
        raise NotImplementedError() 

    # def __call__(self):
    #     # for chaining experiments
    #     raise NotImplementedError()

    @classmethod
    def _get_path_for_result(cls, output_path, run_name, res_name, res_type):
        res_format = cls._get_format_for_type(res_type)
        run_output_dir = cls._get_run_output_dir(output_path, run_name)
        return run_output_dir / f"{res_name}{res_format}"

    @classmethod
    def _has_already_run(cls, output_path, run_name):
        run_dir = cls._get_run_output_dir(output_path, run_name)
        run_dir_exists = run_dir.exists()
        return run_dir_exists

    @classmethod
    def _get_run_output_dir(cls, output_path, run_name):
        return output_path / cls.get_exp_name() / run_name

    @classmethod
    def find_run_names(cls, output_path):
        return list(p.name for p in (output_path / cls.get_exp_name()).iterdir())

    @classmethod
    def clear_missing_results(cls, output_path):
        """Removes empty entries from results"""
        to_remove = []
        for run_name in cls.find_run_names(output_path):
            d = cls._get_run_output_dir(output_path, run_name)
            try:
                next(d.iterdir()) # check if not empty
            except:
                # empty: must be removed
                logging.info("Will remove %d", d)
                to_remove.append(d)

        logging.info("Removing entries...")
        for f in to_remove:
            d.rmdir()

    @classmethod
    def get_exp_name(cls):
        return cls.exp_name

    @staticmethod
    def _get_format_for_type(data_type):
        if issubclass(data_type, pd.DataFrame):
            return EXPERIMENTS_PARAMETERS['extensions']['dataframe']
        if issubclass(data_type, Cascades):
            return EXPERIMENTS_PARAMETERS['extensions']['cascades']
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
    def load_results_for_run(cls, output_path, run_name, datatype=pd.DataFrame, loading_args=None):
        if not cls._has_already_run(output_path, run_name):
            raise Exception(f"No results found in '{output_path}' for run '{run_name}'")
        results = {}
        loading_args = {'index_col': False, **(loading_args or {})}
        for res_name in cls.result_keys:
            filename = cls._get_path_for_result(output_path, run_name, res_name, datatype)
            logging.info("Loading %s", filename)
            results[res_name] = datatype(pd.read_csv(filename, **loading_args))
        return results

    @classmethod
    def load_all_results(cls, output_path, run_col_name='Run', **kwargs):
        run_names = cls.find_run_names(output_path)
        results_collection = (
            (rname, cls.load_results_for_run(output_path, rname))
            for rname in run_names
        )
        return cls.concat_results(results_collection, run_col_name, **kwargs)

    @classmethod
    def concat_results(cls, results_collection: Iterable[Tuple[str, ResultType]], run_col_name='Run', **kwargs):
        """Put together results from several runs,
        adding run name to the index"""
        concat = defaultdict(list)
        for run_name, results in results_collection:
            for res_name, res_df in results.items():
                concat[res_name].append(res_df
                                        .assign(**{run_col_name: run_name})
                                        .set_index(run_col_name, append=True))
        return {
            k: pd.concat(v, sort=True, **kwargs)
            for k, v in concat.items()
        }


class BaseCascadeExperiment(BaseExperiment):

    exp_name = 'cascade_experiment'

    def _load_data(self):
        data_source = self.setup.data_source
        raw_casc = Cascades.from_csv(data_source['doc_path'])
        return {'cascades': raw_casc}

