import pathlib
from dataclasses import dataclass
from typing import Iterable

from models.cascades import Cascades, transform_to_stimulus_response_no_semantic_role
from .common import BaseExperiment, BaseCascadeExperiment, Setup


@dataclass
class StimulusResponse_Setup(Setup):
    pass


class StimulusResponse(BaseCascadeExperiment):

    exp_name = 'stimres'
    setup_class = StimulusResponse_Setup
    result_keys = {'cascades'}

    def _execute(self, data, **kwargs):
        raw_casc: Cascades = data['cascades']
        
        casc = transform_to_stimulus_response(raw_casc)

        return {
            'cascades': casc,
        }


class StimulusResponse_NoSemanticRole(BaseCascadeExperiment):

    exp_name = 'stimres'
    setup_class = StimulusResponse_Setup
    result_keys = {'cascades'}

    def _execute(self, data, **kwargs):
        raw_casc: Cascades = data['cascades']
        
        casc = transform_to_stimulus_response_no_semantic_role(raw_casc)

        return {
            'cascades': casc,
        }


class BaseStimResCascadeExperiment(BaseExperiment):

    exp_name = 'stimres_cascade_experiment'

    def _load_data(self):
        # HOTFIX Yes this is confusing: experiments were supposed to abstract the underlying folder structure
        # This should be refactored
        data_source = self.setup.data_source
        run_path = pathlib.Path(data_source['doc_path'])
        return StimulusResponse.load_results_for_run(output_path=run_path.parent.parent,
                                                     run_name=run_path.name,
                                                     datatype=Cascades,
                                                     loading_args=dict(index_col=[0,1], header=[0,1]))
        #casc = Cascades.from_csv(data_source['doc_path'], index_col=[0,1], header=[0,1])
        #return {'cascades': casc}