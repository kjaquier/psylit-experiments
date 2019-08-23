import spacy.matcher as spmatch
import pandas as pd
import numpy as np

from .dependencies import DEP_INSIDE_SUBSENTENCE


def iter_subsent_roots(doc):
    matcher = spmatch.Matcher(doc.vocab)
    matcher.add('subsent_root', None, [{'DEP': {'NOT_IN': DEP_INSIDE_SUBSENTENCE}}])
    for _, start, end in matcher(doc):
        yield doc[start:end].root
    

def time_remapping(doc):
    subsent_roots = iter_subsent_roots(doc)
    breaks = [sroot.left_edge.i for sroot in subsent_roots]
    idx = pd.IntervalIndex.from_breaks(breaks)
    n_intervals = len(idx)
    return pd.Series(np.arange(n_intervals), index=idx)
