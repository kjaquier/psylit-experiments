"""recnet.py is much faster since it's jit-compiled using numba, 
and also leverages numpy whenever possible"""

import numpy as np
from collections import UserDict, Counter, defaultdict, namedtuple
from itertools import islice
import pandas as pd
import seaborn as sns
import math

try:
    import networkx as nx
except ImportError as e:
    print(e)
    nx = None

try:
    import igraph as ig
except ImportError as e:
    print(e)
    ig = None

class UniqueDict(UserDict):
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def add(self, x):
        try:
            return self.data[x]
        except KeyError:
            n = len(self.data)
            self.data[x] = n
            return n
        
    def to_list(self):
        return sorted(self, key=self.get)
    
    
class RecNet():
    
    def __init__(self, seq, y_proj=None):
        edges = []
        nodes = []
        tokens = []
        token_edges = []
        first_occ = {}
        unique_tokens = UniqueDict()
        seq = map(frozenset, filter(bool, seq))
        ylast = 0
        ypos = []
        for t, tok in enumerate(seq):
            y = None
            tok_id = unique_tokens.get(tok, None)
            if not tok_id:
                tok_id = unique_tokens.add(tok)
                first_occ[tok_id] = t
                
            matched = set()
            ysum = 0
            ywsum = 0
            for d, tok_prev in enumerate(reversed(tokens)):
                recent_common = (tok - matched) & tok_prev
                common = tok & tok_prev
                if not recent_common:
                    continue
                
                w = len(common)
                ywsum += w
                ysum += w * ypos[t-d-1]
                
                edges.append((t-d-1, t, recent_common))
                matched.update(recent_common)
                
                if len(tok) == len(matched):
                    pass
            
            if ysum:
                y = ysum / ywsum + len(tok - matched) * (np.random.random())
            else:
                y = ylast + len(tok - matched) * (1+np.random.random())
                ylast = y
            
            tokens.append(tok)
            nodes.append(tok_id)
            ypos.append(y)
        
        self.nodes = np.array(nodes)
        self.tokens = tokens
        self.unique_tokens = np.array(unique_tokens.to_list())
        self.first_occ = np.array([first_occ[t] for t in range(len(self.unique_tokens))])
        self.weights = [w for _,_,w in edges]
        self.edges = np.array([(t0,t1) for t0,t1,_ in edges])
        self.edges_token_ids = self.nodes[self.edges]
        
        assert np.unique(self.nodes).shape[0] == self.unique_tokens.shape[0]

        if nx:
            self.g = nx.DiGraph()
            self.g.add_nodes_from(np.arange(len(self.nodes)))
            self.g.add_edges_from(self.edges)
        
        if ig:
            self.ig = ig.Graph([(t0,t1) for t0,t1,_ in edges], directed=True)
            
        self.ypos = np.array(ypos)
    
    def entropy(self):
        s = []
        n = self.nodes.shape[0]
        for i in range(n):
            _, counts = np.unique(self.nodes[:i], return_counts=True)
            p = counts / i
            s.append(-np.sum(p * np.log(p)))
        return np.array(s)
    
    def kldivergence(self):
        k = []
        n = self.nodes.shape[0]
        p0 = 1 / n
        for i in range(1, n+1):
            _, counts = np.unique(self.nodes[:i], return_counts=True)
            p = counts / i
            k.append(np.sum(p * np.log(p / p0)))
            p0 = p
        return np.array(k)
    
    def df(self):
        return pd.DataFrame({
            't': np.arange(self.nodes.shape[0]),
            'id': self.nodes,
            'text': [' ; '.join(t) for t in self.tokens],
            'first_occ': self.first_occ[self.nodes],
            'token': self.unique_tokens[self.nodes],
            'indeg': self.indeg(),
            'outdeg': self.outdeg(),
            'entropy': self.entropy(),
            #'kldivergence': self.kldivergence(),
        })
        
    def get_y_proj(self, y_proj=None):
        if y_proj == 'random':
            return np.random.random([len(self.unique_tokens)])
        elif y_proj == 'node_id':
            return self.nodes / len(self.unique_tokens)
        elif y_proj == 'kk':
            #pos_dict = nx.kamada_kawai_layout(self.g, dim=1)
            #return np.array([pos_dict[n][0] for n in self.nodes])
            return np.array(self.ig.layout('kk', dim=2))
        elif y_proj == 'spectral':
            pos_dict = nx.spectral_layout(self.g, dim=1)
            return np.array([pos_dict[n][0] for n in self.nodes])
        elif y_proj == 'prob':
            # TODO place nodes according to their frequency
            raise NotImplementedError
        else:
            return self.ypos
        
    def summary(self, verbose_level=1):
        fs_fmt = lambda fs: "{" + ','.join(fs) + "}"
        list_fmt = lambda xs: "[" + ', '.join(xs) + "]"
        if verbose_level > 1:
            print("cascade identifiers:", list_fmt(fs_fmt(t) for t in self.unique_tokens))
        print(len(self.unique_tokens), "cascade identifiers")
        print(len(self.nodes), "nodes")
        if verbose_level > 1:
            print("edges:", list_fmt(f"{s}-{d} {fs_fmt(w)}" for (s, d), w in zip(self.edges, self.weights)))
        print(len(self.edges), "edges")
        
    def simplify(self):
        def new_seq():
            nodes_ids = np.argsort(self.first_occ)
            yield from (self.tokens[self.nodes[i]] for i in nodes_ids)
        
        return RecNet(t for t in new_seq())
        
    def degdiffs(self):
        Fn = []
        for node, (t, adj) in enumerate(self.g.adjacency()):
            f = np.array(list(adj))
            f[f < t] = +1
            f[f > t] = -1
            f = np.sum(f)
            Fn.append(f)
        Fn = np.array(Fn)
        return Fn
    
    def outdeg(self):
        return np.array([d for _,d in self.g.out_degree])
        #arr = lambda adj: np.array(list(adj))
        #return np.array([
        #    (arr(tj) > ti).sum()
        #    for ti, tj in self.g.adjacency()])
    
    def indeg(self):
        return np.array([d for _,d in self.g.in_degree])
    
    def timedeltas(self):
        return self.edges[:,1] - self.edges[:,0]
    
    def _marker_style(self, Fn=None):
        return dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=Fn or 'black',
            size=4,
            opacity=0.6,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(
                color='white', # would like to put Fe here but not supported :(
                width=0.3,
            ))
    
    def plot3d(self):
        layt = np.array(self.ig.layout('fr3d'))
        Xn = layt[:,0]
        Yn = layt[:,1]
        Zn = layt[:,2]
        E = layt[self.edges]
        #Xe = [[e0, e1, None] for e0, e1 in E[:,:,0]]
        Xe = E[:,:,0].flatten()
        Ye = E[:,:,1].flatten()
        Ze = E[:,:,2].flatten()
        #Ye = [[e0, e1, None] for e0, e1 in E[:,:,1]]
        #Ze = [[e0, e1, None] for e0, e1 in E[:,:,2]]
        print(layt[:5])
        print(Xe[:5])
        
        Fn = None#self.degdiffs()
        
        labels = [";".join(sorted(t)) for t in self.tokens]
        
        edge_trace=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgba(227, 229, 232, 0.1)', width=.5),
               #hoverinfo='none'
               )

        node_trace=go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers',
                       #name='actors',
                       marker=self._marker_style(Fn),
                       text=labels,
                       hoverinfo='text'
                       )
        
        axis=dict(showbackground=False,
                  showline=False,
                  zeroline=False,
                  showgrid=False,
                  showticklabels=False,
                  title=''
                  )

        layout = go.Layout(
            title="Recurrence Network",
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            margin=dict(
                t=100
            ),
            hovermode='closest',
        )
        
        fig = go.Figure(data=[edge_trace,node_trace], layout=layout)
        
        plotly.offline.iplot(fig)
        
    def plot(self, layout=None, show_labels=False, timeslice=None):
        Xn = np.arange(len(self.nodes))
        Yn = self.get_y_proj(layout)
        Xe = self.edges.flatten()
        Ye = Yn[self.edges_token_ids].flatten()
        
        edge_trace = go.Scattergl(
            x=Xe,
            y=Ye,
            line=dict(color='rgba(125, 125, 125, 0.4)', width=.8),
            mode='lines' + ('+text' if show_labels else ''),
        )
        
        Fn = None#self.degdiffs()
        Fe = self.timedeltas()
        
        node_trace = go.Scattergl(
            x=Xn,
            y=Yn[self.nodes],
            text=["\n".join(sorted(t)) for t in self.tokens],
            mode='markers' + ('+text' if show_labels else ''),
            hoverinfo='text',
            textposition='top center',
            marker=self._marker_style(Fn),
        ) 

        layout = dict(
            title='Recurrence network',
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(step='all')
                    ]
                ),
                rangeslider=dict(
                    visible = True,
                ),
                type='linear',
                range=[0,50],
            ),
        )
        
        fig = go.Figure(data=[edge_trace,node_trace], layout=layout)
        
        plotly.offline.iplot(fig)#, filename='basic-line')

        
arr_eq = np.array_equal#lambda xs, ys: all(x == y for x, y in zip(xs, ys))
        
testnet = RecNet(["AB","BC","CD","ABC","BC","BD"])
assert arr_eq(testnet.nodes,     [0,1,2,3,1,4]), testnet.nodes
assert arr_eq(testnet.first_occ, [0,1,2,3,5]), testnet.first_occ
assert arr_eq(testnet.edges,                np.array([[0,1],[1,2],[2,3],[1,3],[0,3],[3,4],[4,5],[2,5]])), testnet.edges
assert arr_eq(testnet.weights, [frozenset(w) for w in [ "B",  "C",  "C",  "B",  "A", "BC",  "B", "D"]]), testnet.weights
assert arr_eq(testnet.edges_token_ids,      np.array([[0,1],[1,2],[2,3],[1,3],[0,3],[3,1],[1,4],[2,4]])), testnet.edges_token_ids

#testnet = testnet.simplify()
#assert arr_eq(testnet.nodes, [0,1,2,3,4]), testnet.nodes
#assert arr_eq(testnet.edges,                np.array([[0,1],[1,2],[2,3],[1,3],[0,3],[2,4]])), testnet.edges
#assert arr_eq(testnet.weights, [frozenset(w) for w in [ "B",  "C",  "C",  "B",  "A", "D"]]), testnet.weights
#assert arr_eq(testnet.edges_token_ids, testnet.edges), testnet.edges_token_ids
