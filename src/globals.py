import joblib

from parameters import CACHE_PARAMETERS

MEMORY = joblib.Memory(cachedir=CACHE_PARAMETERS['dir'])
