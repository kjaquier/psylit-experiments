from dataclasses import dataclass
from typing import Iterable

from models.cascades import Cascades, FEATURE_TRANSFORMERS
from models.info_dynamics import fast_block_entropy
from .common import BaseCascadeExperiment, Setup


@dataclass
class BlockEntropy_StimulusResponse_Setup(Setup):
    k_values: Iterable[int]
    measure_name: str
    window_size: int = 1


class BlockEntropy_StimulusResponse(BaseCascadeExperiment):

    exp_name = 'blockent_stimres'
    setup_class = BlockEntropy_StimulusResponse_Setup
    result_keys = {'persubj'}

    def _execute(self, data, **kwargs):
        raw_casc: Cascades = data['cascades']
        
        transform_function = FEATURE_TRANSFORMERS['StimulusResponse']
        casc = transform_function(raw_casc)
        
        be = casc.batch_single_measure(
            trajectory_group='Subject',
            measure=fast_block_entropy, 
            measure_name=self.setup.measure_name,
            k_values=self.setup.k_values, 
            local=False,
            window_size=self.setup.window_size,
        )

        return {
            'persubj': be,
        }
