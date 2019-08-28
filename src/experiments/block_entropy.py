import dataclasses as dc
from typing import Dict, Tuple

import pandas as pd
import pyinform

from .common import BaseCascadeExperiment, ExperimentSetup
from models.cascades import Cascades, FEATURE_TRANSFORMERS
from models.cascades import Cascades
from utils.misc import path_remove_if_exists
from parameters import ANALYSIS_PARAMETERS


class BlockEntropy_StimulusResponse(BaseCascadeExperiment):

    exp_name = 'blockent_stimres'

    def _execute(self, data):
        measure = 'block_entropy'
        raw_casc = data['cascades']
        
        transform_function = FEATURE_TRANSFORMERS['StimulusResponse']
        casc = transform_function(raw_casc)
        
        B = casc.batch_single_measure(
            trajectory_group='Subject',
            measure=pyinform.blockentropy.block_entropy, 
            measure_name=ANALYSIS_PARAMETERS[measure]['name'],
            k=ANALYSIS_PARAMETERS[measure]['k_values'], 
            local=False,
        )