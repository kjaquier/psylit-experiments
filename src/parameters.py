import pathlib
import sys
import logging


LOGGING_PARAMETERS = dict(
    level=logging.INFO, 
    stream=sys.stdout, 
    format='[%(asctime)s %(levelname)s %(name)s (%(funcName)s)] %(message)s',
    datefmt='%H:%M:%S',
)

PREPARE_PARAMETERS = dict(
    batch_size=20000,
    spacy_model='en_core_web_sm', 
    coref=dict(
        max_dist=50,
        max_dist_match=500,
        greedyness=0.5,
    ),
    lexicon=dict(
        threshold=lambda series: series.mean(),
    ),
    extensions=dict(
        data='.data.csv.zip',
        entities='.ent.csv.zip',
        features='.feat.csv.zip',
        metadata='.meta.json',
    ),
)

PROCESS_PARAMETERS = dict(
    min_entities_occurrences=100,
    extensions=dict(
        cascades='.csv.zip',
        data_input='.data.csv*',
        entities_input='.ent.csv*',
        **PREPARE_PARAMETERS['extensions'],
    ),
)

ANALYSIS_PARAMETERS = dict(
    block_entropy=dict(
        name='$H(k)$',
        extension='.be.csv',
        k_values=range(1, 20, 3),
        transformer='StimulusResponse',
        local=False,
    ),
)

CACHE_PARAMETERS = dict(
    dir=pathlib.Path() / 'cache',
)
