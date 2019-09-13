"""Sources:
https://gist.github.com/dglmoore/0751b3023d082a44de775d5c83a009b9
https://github.com/ELIFE-ASU/Inform/issues/83
"""

import numpy as np
from pyinform.utils.coalesce import coalesce_series
from pyinform import block_entropy, transfer_entropy, active_info, entropy_rate
from numba import jit, int8, int32, int64
from numba.typed import Dict
import pandas as pd
import pytest


# @jit
# def coalesce_history_jit(series: np.int8, k):
#     series = series.astype(np.int64) - np.min(series)
#     b: int64 = np.max(series) + 1
#     space_mapping = Dict.empty(
#         key_type=int64,
#         value_type=int32,
#     )
#     states = np.empty(len(series) - k + 1, dtype=np.int64)
#     histories = np.empty(len(series) - k + 1, dtype=np.int32)
#     last_state_image: int32 = 0
#     state: int64 = 0
#     q: int64 = 1
#     for i in range(k-1):
#         q *= b
#         state = state * b + series[i]

#     for i, j in enumerate(range(k-1, len(series))):
#         state = state * b + series[j]
#         states[i] = state
#         # histories[i] = space_mapping.get(state, last_state_image)
#         # if histories[i] == last_state_image:
#         #     space_mapping[state] = last_state_image
#         #     last_state_image += 1
#         state = state - q * series[j - k + 1]

#     unique_states, histories = np.unique(states, return_inverse=True)

#     return histories



def coalesce_history(series, k):
    pows = 2**np.arange(k)
    black_box = lambda h: pows[h.astype(np.bool)].sum()
    bb = series.rolling(k, min_periods=k).apply(black_box, raw=True)
    #_, coalesced = np.unique(bb.values, return_inverse=True)
    coalesced, base = coalesce_series(bb)
    return coalesced[k-1:]

def test_coalesce():
    x = np.array([0, 0, 1, 1, 0, 1]).astype(np.int8)
    series = pd.Series(x)
    assert coalesce_history(series, 1).shape == x.shape
    assert coalesce_history(series, 2).shape == np.array([0, 1, 2, 3, 1]).shape
    assert coalesce_history(series, 3).shape == np.array([0, 1, 2, 3]).shape
    assert coalesce_history(series, 4).shape == np.array([0, 1, 2]).shape
    assert coalesce_history(series, 5).shape == np.array([0, 1]).shape
    # np.testing.assert_equal(coalesce_history(series, 1), series)
    # np.testing.assert_equal(coalesce_history(series, 2), np.array([0, 1, 2, 3, 1]).astype(np.int64))
    # np.testing.assert_equal(coalesce_history(series, 3), np.array([0, 1, 2, 3]).astype(np.int64))
    # np.testing.assert_equal(coalesce_history(series, 4), np.array([0, 1, 2]).astype(np.int64))
    # np.testing.assert_equal(coalesce_history(series, 5), np.array([0, 1]).astype(np.int64))

L = 10000


def test_coalesce_block_entropy():
    # H(x^k) ~ H_k(x)
    x = (np.random.random([L]) > .5).astype(np.int8)
    series = pd.Series(x)
    for k in range(1, 29):
        history_series = coalesce_history(series, k)
        assert np.isclose(block_entropy(history_series, k=1), block_entropy(x, k=k))

    # Doesn't overflow as early
    history_series = coalesce_history(series, 50)
    block_entropy(history_series, k=1)

# def test_coalesce_active_info():
#     x = (np.random.random([L]) > .5).astype(np.int8)
#     series = pd.Series(x)
#     for k in range(1, 29):
#         history_series = coalesce_history(series, k)
#         assert np.isclose(active_info(history_series, k=1), active_info(x, k=k))

    # Doesn't overflow as early
    history_series = coalesce_history(series, 50)
    active_info(history_series, k=1)

def test_coalesce_entropy_rate():
    x = (np.random.random([L]) > .5).astype(np.int8)
    series = pd.Series(x)
    for k in range(1, 29):
        history_series = coalesce_history(series, k)
        assert np.isclose(entropy_rate(history_series, k=1), entropy_rate(x, k=k))

    # Doesn't overflow as early
    history_series = coalesce_history(series, 50)
    entropy_rate(history_series, k=1)


def test_coalesce_transfer_entropy():
    # T(x^k->y^k) ~ T_k(x->y)
    x = (np.random.random([L]) > .5).astype(np.uint8)
    y = np.logical_xor(x, (np.random.random([L]) > .5)).astype(np.uint8)
    for k in range(1, 29):
        xc = coalesce_history(x, k)
        yc = coalesce_history(y, k)
        assert np.isclose(transfer_entropy(xc, y[:L-k+1], k=1), transfer_entropy(x, y, k=k), atol=1.e-3)
        assert np.isclose(transfer_entropy(yc, x[:L-k+1], k=1), transfer_entropy(y, x, k=k), atol=1.e-3)

    # Doesn't overflow as early
    for k in range(1, 100):
        history_series = coalesce_history(series, 100)
        transfer_entropy(history_series, k=1)

    # for k in range(29, 50):
    #     history_series = coalesce_history(series, k)
    #     print(k, transfer_entropy(history_series, k=1))


if __name__ == "__main__":
    test_coalesce_block_entropy()
    test_coalesce_transfer_entropy()