import logging
from dataclasses import dataclass
from typing import Optional, List

import pyinform

from models.cascades import Cascades, FEATURE_TRANSFORMERS
import models.info_dynamics as info_dynamics
from .common import BaseCascadeExperiment, Setup


logger = logging.getLogger()


@dataclass
class TransferEntropy_StimulusResponse_Setup(Setup):
    k: int
    variant: str = 'apparent'
    window_size: int = 1
    src_cols: Optional[List[str]] = None
    dest_cols: Optional[List[str]] = None


@dataclass
class CompleteTransferEntropy_StimulusResponse_Setup(Setup):
    k: int
    window_size: int = 1
    cols: Optional[List[str]] = None
    min_p_value: float = 0.05


class CompleteTransferEntropy_StimulusResponse(BaseCascadeExperiment):

    exp_name = 'ctfrent_stimres'
    setup_class = CompleteTransferEntropy_StimulusResponse_Setup
    result_keys = {'persubj'}

    def _execute(self, data, **kwargs):
        raw_casc: Cascades = data['cascades']
        
        transform_function = FEATURE_TRANSFORMERS['StimulusResponse']
        casc = transform_function(raw_casc)

        get_args = lambda col: {
            'k': self.setup.k,
            'min_p_value': self.setup.min_p_value,
        }

        cols = self.setup.cols or [
            c for c in casc.casc.columns if c in self.setup.src_cols
        ]

        res = casc.batch_multi_df_measure(trajectory_group='Subject',
                                          measure=info_dynamics.multi_complete_tfr_entropy,
                                          cols=cols,
                                          window_size=self.setup.window_size,
                                          get_args=get_args,
                                          )

        return {
            'persubj': res.reset_index(),
        }


class TransferEntropy_StimulusResponse(BaseCascadeExperiment):

    exp_name = 'tfrent_stimres'
    setup_class = TransferEntropy_StimulusResponse_Setup
    result_keys = {'persubj'}

    def _execute(self, data, **kwargs):
        raw_casc: Cascades = data['cascades']
        
        transform_function = FEATURE_TRANSFORMERS['StimulusResponse']
        casc = transform_function(raw_casc)

        measure = {
            'apparent': info_dynamics.apparent_tfr_entropy,
            'cond': info_dynamics.cond_tfr_entropy,
        }[self.setup.variant]

        get_args = lambda src, dst: {
            'k': self.setup.k
        }

        src_cols = self.setup.src_cols or [
            c for c in casc.casc.columns if c in self.setup.src_cols
        ]
        dest_cols = self.setup.dest_cols or [
            c for c in casc.casc.columns if c in self.setup.dest_cols
        ]

        res = casc.batch_pairwise_df_measure(trajectory_group='Subject',
                                             measure=measure,
                                             src_cols=src_cols,
                                             dest_cols=dest_cols,
                                             window_size=self.setup.window_size,
                                             get_args=get_args,
                                             )

        return {
            'persubj': res.reset_index(),
        }
