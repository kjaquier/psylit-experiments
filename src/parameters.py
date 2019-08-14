import sys
import logging


LOGGING_PARAMETERS = dict(
    level=logging.DEBUG, 
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
    )
)

PROCESS_PARAMETERS = dict(
    min_entities_occurrences=100,
    input_file_patterns=dict(
        data='*.data.csv',
        entities='*.ent.csv',
        features='*.feat.csv',
        meta='*.meta.json',
    )
)
