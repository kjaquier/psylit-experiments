from dataclasses import dataclass
from typing import Iterable

import pyinform

from .common import BaseCascadeExperiment, Setup
from models.cascades import Cascades, FEATURE_TRANSFORMERS
from utils.misc import path_remove_if_exists


@dataclass
class BlockEntropy_StimulusResponse_Setup(Setup):
    k_values: Iterable[int]
    measure_name: str


class BlockEntropy_StimulusResponse(BaseCascadeExperiment):

    exp_name = 'blockent_stimres'
    setup_class = BlockEntropy_StimulusResponse_Setup
    result_keys = {'persubj'}

    def _execute(self, data):
        raw_casc: Cascades = data['cascades']
        
        transform_function = FEATURE_TRANSFORMERS['StimulusResponse']
        casc = transform_function(raw_casc)
        
        be = casc.batch_single_measure(
            trajectory_group='Subject',
            measure=pyinform.blockentropy.block_entropy, 
            measure_name=self.setup.measure_name,
            k_values=self.setup.k_values, 
            local=False,
        )

        return {
            'persubj': be,
        }