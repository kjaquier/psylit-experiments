import logging
from collections import defaultdict
from itertools import product
from functools import partial
from utils.jvm import infodynamics_measures_discrete
from utils.java import df_as_java_array
from utils.misc import trace, cached, HashableDict
from parameters import CACHE_PARAMETERS

import numpy as np
import pandas as pd
import pyinform

logger = logging.getLogger()

TransferEntropyDiscrete = infodynamics_measures_discrete.TransferEntropyCalculatorDiscrete
CondTransferEntropyDiscrete = infodynamics_measures_discrete.ConditionalTransferEntropyCalculatorDiscrete


@cached
def coalesce_history(series, k):
    pows = 2**np.arange(k)
    black_box = lambda h: pows[h.astype(np.bool)].sum()
    bb = series.rolling(k, min_periods=k).apply(black_box, raw=True)
    #_, coalesced = np.unique(bb.values, return_inverse=True)
    coalesced, base = pyinform.utils.coalesce_series(bb)
    return coalesced[k-1:], base



@cached
def fast_block_entropy(series, k, **kwargs):
    series_c, base = coalesce_history(series, k)
    be = pyinform.block_entropy(series_c, 1, **kwargs)
    # er = pyinform.entropy_rate(series_c, 1, **kwargs)

    return {'n': len(series_c),
            'k': k,
            'be': be,
            # 'er': er,
            }


@cached
def apparent_tfr_entropy(casc, src_col, dst_col, k, n_permutations=100, base=2):
    te_calc = TransferEntropyDiscrete(base, k)
    te_calc.initialise()
    
    source_series = df_as_java_array(casc, src_col)
    dest_series = df_as_java_array(casc, dst_col)
    te_calc.addObservations(source_series, dest_series)
    
    meas = te_calc.computeAverageLocalOfObservations()
    surrogate_dist_an = te_calc.computeSignificance() # analytic distribution of measurement under null hypothesis
    p_val_an = surrogate_dist_an.pValue # analytic probability that the surrogate is greater than the measurement (i.e measurement non-significant)
    surrogate_dist = te_calc.computeSignificance(n_permutations) # distribution of measurement under null hypothesis
    p_val = surrogate_dist.pValue # probability that the surrogate is greater than the measurement (i.e measurement non-significant)

    return {'n': len(source_series),
            'k': k,
            'apparent_te': meas,
            'apparent_te_p_value': p_val,
            'apparent_te_p_value_analytical': p_val_an}


@cached
def cond_transfer_entropy(casc, src_col, dst_col, cond_cols, k, n_permutations=100, base=2):
    cond_cols = [src_col]#list(cond_cols)
    num_contributors = len(cond_cols)
    te_calc = CondTransferEntropyDiscrete(base, k, num_contributors)
    te_calc.initialise()

    source_series = df_as_java_array(casc, src_col)
    dest_series = df_as_java_array(casc, dst_col)
    conds_series = df_as_java_array(casc, cond_cols)
    te_calc.addObservations(source_series, dest_series, conds_series)

    meas = te_calc.computeAverageLocalOfObservations()
    surrogate_dist_an = te_calc.computeSignificance() # analytic distribution of measurement under null hypothesis
    p_val_an = surrogate_dist_an.pValue # analytic probability that the surrogate is greater than the measurement (i.e measurement non-significant)
    surrogate_dist = te_calc.computeSignificance(n_permutations) # distribution of measurement under null hypothesis
    p_val = surrogate_dist.pValue # probability that the surrogate is greater than the measurement (i.e measurement non-significant)

    return {'n': len(source_series),
            'k': k,
            'cond_te': meas,
            'cond_te_p_value': p_val,
            'cond_te_p_value_analytical': p_val_an}


def multi_complete_tfr_entropy(df, cols, k, n_permutations=100, base=2, min_p_value=0.05):
    num_contributors = len(df.columns.values) - 2
    assert num_contributors >=1
    series = {c: df_as_java_array(df, c) for c in cols}

    te_args = {'k': k, 'n_permutations': n_permutations, 'base': base}

    te = partial(apparent_tfr_entropy, df, **te_args)
    cte = partial(cond_transfer_entropy, df, **te_args)
    
    cols = set(cols)
    parents = {}
    for var in cols:
        var_parents = new_parents = {
            c for c in cols
            if c != var and te(c, var)['apparent_te_p_value'] < min_p_value
        }
        while new_parents:
            candidates = cols - var_parents - {var}
            logger.debug('%s: %d parent(s), %d candidate(s)', var, len(var_parents), len(candidates))
            new_parents = {
                c for c in candidates
                if cte(c, var, var_parents)['cond_te_p_value'] < min_p_value
            }
            var_parents |= new_parents
        parents[var] = frozenset(var_parents)

    results = {
        (c, var, parents[var]): cte(c, var, parents[var])
        for var in cols
        for c in cols
        if c != var
    }

    return [
        {'n': len(source_series),
         'k': k,
         'Source': c,
         'Target': var,
         'complete_te': te,
         'complete_te_p_value': p,
         **{p: (p in parents) for p in cols}}
        for (c, var, parents), (te, p) in results.items()
    ]
