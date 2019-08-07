

from statistics import mean

from os import linesep as EOL

import re
import json

import pandas as pd
import numpy as np


class MultiCascade:

    def __init__(self, data):
        self.data = data
    
    def pipe(self, *args, **kwargs):
        return MultiCascade(self.data.pipe(*args, *kwargs))

    def filter(self, mask):
        return MultiCascade(self.data[mask])

    def filter_any(self, cols=None):
        mask = (self.data[cols] if cols else self.data).any()
        return self.filter(mask)

    def filter_all(self, cols=None):
        mask = (self.data[cols] if cols else self.data).all()
        return self.filter(mask)

    def zeros(self, cols):
        mask = (self.data[cols] if cols else self.data) < 1
        return self.filter(mask)

    def merge_cols(self, cols, merge_func):
        return
    