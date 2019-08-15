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
    coref=dict(
        max_dist=50,
        max_dist_match=500,
        greedyness=0.5,
    ),
    lexicon=dict(
        threshold=lambda series: series.mean(),
    ),
)

PROCESS_PARAMETERS = dict(
    min_entities_occurrences=100,
)

CACHE_PARAMETERS = dict(
    dir=pathlib.Path() / 'cache',
)
