from dataclasses import dataclass
from typing import Optional, List

import pyinform

from models.cascades import Cascades, FEATURE_TRANSFORMERS
from .common import BaseCascadeExperiment, Setup


@dataclass
class TransferEntropy_StimulusResponse_Setup(Setup):
    k: int
    conditional: bool
    measure_name: str
    window_size: int = 1
    src_cols: Optional[List[str]] = None
    dest_cols: Optional[List[str]] = None


def get_cols_at_2nd_level(df, cols):
    for first_lvl_value in df.index.get_level_values(0):
        for col in cols:
            yield first_lvl_value, col


class TransferEntropy_StimulusResponse(BaseCascadeExperiment):

    exp_name = 'tfrent_stimres'
    setup_class = TransferEntropy_StimulusResponse_Setup
    result_keys = {'persubj'}

    def _execute(self, data, **kwargs):
        raw_casc: Cascades = data['cascades']
        
        transform_function = FEATURE_TRANSFORMERS['StimulusResponse']
        casc = transform_function(raw_casc)

        # TODO use condition
        get_args = lambda src, dst: {
            'k': 1
        }
        src_cols = get_cols_at_2nd_level(casc.casc, self.setup.src_cols) if self.setup.src_cols else None
        dest_cols = get_cols_at_2nd_level(casc.casc, self.setup.dest_cols) if self.setup.src_cols else None
        te = casc.batch_pairwise_measure(trajectory_group='Subject',
                                         measure=pyinform.transferentropy.transfer_entropy,
                                         measure_name=self.setup.measure_name,
                                         src_cols=src_cols,
                                         dest_cols=dest_cols,
                                         window_size=self.setup.window_size,
                                         get_args=get_args,
                                         )

        return {
            'persubj': te,
        }
