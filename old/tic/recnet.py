import numpy as np
from collections import UserDict, Counter, defaultdict, namedtuple
from itertools import islice
import pandas as pd
import seaborn as sns
import math

try:
    import numba
    import numba.typed
    import numba.types
    numba_on = True
except ImportError as e:
    print(e)
    NumbaDummy = namedtuple('NumbaDummy', 'jit')
    numba = NumbaDummy(jit=lambda f: f)
    numba_on = False

InfoCascades = namedtuple('InfoCascades', [
    'nodes',
    'edges',
    'indeg',
    'outdeg',
    'n_nodes',
    'n_edges',
    'n_events',
    'n_microtokens',
    'w',
    'time_deltas',
    'event_size',
    'event_seq',
    'events',
    'events_first',
    'events_last',
    'events_count',
    'microtokens',
    'microtokens_first',
    'microtokens_last',
    'microtokens_count'])

if numba_on:
    def OccurencesDict():
        return numba.typed.Dict.empty(
            key_type=numba.types.unicode_type,
            value_type=numba.types.int32[3], # 0=first, 1=last, 2=count
        )
else:
    OccurencesDict = dict

@numba.jit
def make_occurences():
    return OccurencesDict()

@numba.jit
def update_occurences(occs, t, key):
    if key in occs:
        first, _last, count = occs[key]
        occs[key] = np.array([first, t, count+1])
    else:
        occs[key] = np.array([t, t, 1])
    return occs
        
@numba.jit
def make_info_cascades(event_list):
    n = len(event_list)
    
    edges = []
    ev_occs = make_occurences()
    mtoks_occs = make_occurences()
    sizes = np.zeros([n])
    ev_seq = np.zeros([n])
    
    for t, ev in enumerate(event_list):
        ev_str = str(hash(ev))
        update_occurences(ev_occs, t, ev_str)
        sizes[t] = len(ev)
        ev_seq[t] = ev_occs[ev_str][1]
        for mtok in ev:
            occs = mtoks_occs.get(mtok, None)
            if occs is not None:
                mtok_last = occs[1]
                if mtok_last != t:
                    if edges and mtok_last == edges[-1][0]:
                        edges[-1][2] += 1
                    else:
                        edges.append([mtok_last,t,1,t-mtok_last])
            update_occurences(mtoks_occs, t, mtok)
    
    edges = np.array(edges)
    w = edges[:,2]
    time_deltas = edges[:,3]
    edges = edges[:,:2]
    
    n_evs = len(ev_occs)
    n_mtoks = len(mtoks_occs)
    
    evs = [""] * n_evs
    ev_first = np.zeros([n_evs])
    ev_last = np.zeros([n_evs])
    ev_count = np.zeros([n_evs])
    mtoks = [""] * n_mtoks
    mtok_first = np.zeros([n_mtoks])
    mtok_last = np.zeros([n_mtoks])
    mtok_count = np.zeros([n_mtoks])
    
    i = 0
    for k, ev in ev_occs.items():
        first, last, count = ev
        evs[i] = k
        ev_last = last
        ev_count = count
        i += 1
    
    i = 0
    for k, mtok in mtoks_occs.items():
        first, last, count = mtok
        mtoks[i] = k
        mtok_first = first
        mtok_last = last
        mtok_count = count
        i += 1
        
    indeg = np.bincount(edges[:,1])
    pad1 = np.zeros(n - indeg.shape[0])
    indeg = np.concatenate([indeg, pad1])
    
    outdeg = np.bincount(edges[:,0])
    pad2 = np.zeros(n - outdeg.shape[0])
    outdeg = np.concatenate([outdeg, pad2])
        
    return InfoCascades(
        nodes=np.arange(n),
        edges=np.array(edges),
        indeg=indeg,
        outdeg=outdeg,
        n_nodes=n,
        n_edges=edges.shape[0],
        n_events=n_evs,
        n_microtokens=n_mtoks,
        w=w,
        event_seq=ev_seq,
        events=evs,
        event_size=sizes,
        time_deltas=time_deltas,
        events_first=ev_first,
        events_last=ev_last,
        events_count=ev_count,
        microtokens=mtoks,
        microtokens_first=mtok_first,
        microtokens_last=mtok_last,
        microtokens_count=mtok_count,
    )

@numba.jit
def get_ypos(c):
    firsts = c.event_size - c.outdeg  # number of first occurences per node
    y = np.cumsum(firsts)
    
    for i in c.nodes:
        is_parent = c.edges[:,1] == i
        parents_edges = c.edges[is_parent]
        parents = parents_edges[:,0]
        parents_y = y[parents]
        parents_weight = c.w[is_parent]
        wsum = parents_weight.sum()
        if wsum:
            y[i] = (parents_weight * parents_y).sum() / wsum
        
    return y

class RecNet:
    
    def __init__(self, seq, y_proj=None):
        self.seq = [tuple(toks) for toks in seq]
        self._c = make_info_cascades(self.seq)
        # self.nodes = self._c.nodes
        # self.edges = self._c.edges
        # self.weights = self._c.w
        # self.time_deltas = self._c.time_deltas
        # self.first_occ = self._c.microtokens_first
        
    def get_y_proj(self, layout=None):
        return get_ypos(self._c)
    
    def _txt_seq(self):
        return [" ; ".join(sorted(t)) for t in self.seq]

    def cascades_layout(self, layout=None):
        Xn = np.arange(self._c.n_nodes)
        Yn = self.get_y_proj(layout)
        Xe = self._c.edges.flatten()
        Ye = Yn[Xe]
        return {
            'nodes': {
                'x': Xn,
                'y': Yn,
                'text': self._txt_seq(),
            },
            'edges': {
                'x': Xe,
                'y': Ye,
                'text': None,
            },
        }

    def entropy(self):
        s = []
        n = self._c.n_nodes
        for i in range(n):
            _, counts = np.unique(self._c.event_seq[:i], return_counts=True)
            p = counts / i
            s.append(-np.sum(p * np.log(p)))
        return np.array(s)
    
    def df(self):
        return ({
            't': np.arange(self._c.n_events),
            'text': self._txt_seq(),
            'first_occ': self._c.events_first,
            'last_occ': self._c.events_last,
            'token': self._c.events,
            'indeg': self._c.indeg,
            'outdeg': self._c.outdeg,
            'entropy':self.entropy(),
            #'kldivergence': self.kldivergence(),
        }), 

if __name__ == "__main__":
        
    arr_eq = np.array_equal#lambda xs, ys: all(x == y for x, y in zip(xs, ys))
            
    testnet = RecNet(["AB","BC","CD","ABC","BC","BD"])
    testnet.get_y_proj()

    # assert arr_eq(testnet.nodes,     [0,1,2,3,1,4]), testnet.nodes
    # assert arr_eq(testnet.first_occ, [0,1,2,3,5]), testnet.first_occ
    # assert arr_eq(testnet.edges,                np.array([[0,1],[1,2],[2,3],[1,3],[0,3],[3,4],[4,5],[2,5]])), testnet.edges
    # assert arr_eq(testnet.weights, [frozenset(w) for w in [ "B",  "C",  "C",  "B",  "A", "BC",  "B", "D"]]), testnet.weights
    # assert arr_eq(testnet.edges_token_ids,      np.array([[0,1],[1,2],[2,3],[1,3],[0,3],[3,1],[1,4],[2,4]])), testnet.edges_token_ids
    print("tests passed")