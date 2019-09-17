from dataclasses import dataclass
from typing import Iterable

from models.cascades import Cascades
from models.info_dynamics import fast_block_entropy
from .common import Setup
from .stim_res import BaseStimResCascadeExperiment


@dataclass
class BlockEntropy_StimulusResponse_Setup(Setup):
    k_values: Iterable[int]
    measure_name: str
    window_size: int = 1


class BlockEntropy_StimulusResponse(BaseStimResCascadeExperiment):

    exp_name = 'blockent_stimres'
    setup_class = BlockEntropy_StimulusResponse_Setup
    result_keys = {'persubj'}

    def _execute(self, data, **kwargs):
        casc: Cascades = data['cascades']
        
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
