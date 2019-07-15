import logging
import collections as coll
import functools as ft
import numpy as np
import pandas as pd
import networkx as nx
from spacy.tokens import Doc, Span, Token
from spacy.matcher import Matcher
    
def ilen(gen):
    return sum(1 for _ in gen)

AGENT_DEPS = ('nsubj', 'csubj', 'poss', 'expl')
PATIENT_DEPS = ('nsubjpass','csubjpass', 'obj','pobj','dobj','iobj','auxpass','nmod')
DEP_WHITELIST = AGENT_DEPS + PATIENT_DEPS + (
                 'conj','compound','neg','poss',
                 'prep','amod','attr','acl','advcl',
                 'appos','aux','dislocated','obl',
                 'orphan', # connects agent/patient in the conj of a previous predicate
                 # clearly excluded:
                 # 'npadvmod', 'advmod', 'parataxis', 'xcomp'
                 )

PERSON, ENV, UNK, NARR, READ, NAN = ['PERSON', 'ENVIRONMENT', 'UNKNOWN', 'NARRATOR', 'READER', 'NA']

exceptions = {
    ENV:     {'it','this','that','its','itself','something'},
    UNK:   {'they','them','themselves','their','these','those'},
    NARR:    {'i',  'me' ,'myself',   'my',   'mine' },
    READ:    {'you'      ,'yourself', 'your', 'yours'},
    PERSON:  {'he', 'him','himself', 'his', 
              'she','her','herself',          'hers'},
}
exceptions = {w:k for k, v in exceptions.items() for w in v}


def doc_at(doc, indices):
    for i in indices:
        yield doc[i]


class LoggerMixin:

    def __init__(self, logger=None, default_lvl=logging.DEBUG):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(default_lvl)
        self.default_lvl = default_lvl

    def _log(self, msg, *args, **kwargs):
        self.logger.log(self.default_lvl, f"[{self.__class__.__name__}] {msg}", *args, **kwargs)

class DocGraph(LoggerMixin):

    def __init__(self, doc, keep_deps=DEP_WHITELIST, logger=None):
        super().__init__(logger)
        g = nx.DiGraph()

        ents = {m.root.i: clust.main 
                for clust in doc._.coref_clusters 
                for m in clust.mentions}

        matcher = Matcher(doc.vocab)
        matcher.add('edges', None, [{'DEP': {'IN': keep_deps}}])
        
        matches = matcher(doc)
        self._log(f"{len(matches)} matches")
        edges = []
        for _, start, end in matches:
            tok = doc[start:end].root
            head = tok.head # if root then src.head == src
            
            # those relations should be reversed when propagating the agent
            if (tok.dep_ in AGENT_DEPS) or (tok.dep_ == 'conj' and head.i in ents):
                tok, head = head, tok
                
            # dep tree goes from leaf to root, now we go from root to leaf
            edges.append((head.i, tok.i, head.dep_, tok.dep_))
        
        # remove all incoming edges for roots other than agent relations (will be the entry points)
        A = pd.DataFrame(edges, columns=['src','dst','src_dep','dst_dep'])
        roots = set(A[(A.dst_dep == 'ROOT') | A.src_dep.isin(AGENT_DEPS)].dst.unique())
        dst_is_root = A.dst.isin(roots)
        A = A[dst_is_root & A.src_dep.isin(AGENT_DEPS) | ~dst_is_root]
        A.pop('src_dep')
        A.pop('dst_dep')
        
        g.add_edges_from(A.itertuples(index=False))
        self._log(f"{nx.number_of_edges(g)} edges, {nx.number_of_nodes(g)} nodes")

        # connect coreferences by their root
        mentions_edges = []
        for clust in doc._.coref_clusters:
            c_i = clust.main.root.i
            for mention in clust.mentions:
                m_i = mention.root.i
                mentions_edges.append((c_i, m_i))
        g.add_edges_from(mentions_edges)

        self._log(f"{nx.number_of_edges(g)} edges, {nx.number_of_nodes(g)} nodes")

        nodes = pd.DataFrame(((n.i, n.i in ents, n._.has_lex, None) 
                            for n in doc_at(doc, g.nodes)), 
                            columns=('i','is_ent','is_matched','categ'))
        nodes.set_index('i', inplace=True)
        nodes.loc[nodes.is_matched,'categ'] = 'Matched'
        nodes.loc[nodes.is_ent,'categ'] = 'Entity'

        self._log(nodes.categ.value_counts(dropna=False))

        self.graph = g
        self.categ = nodes.categ
        self.doc = doc
        self.ents = ents

    def iter_frames(self):
        Frame = coll.namedtuple('Frame', 't,predicate,agent,patient')
        doc = self.doc
        categ = self.categ
        ents = self.ents
        curr_t = -1
        curr_predicate = None
        for clust in doc._.coref_clusters:
            ent = clust.main
            for src, dst, edge_type in nx.dfs_labeled_edges(self.graph, source=ent.root.i):
                if edge_type == 'forward':
                    dst_tok = doc[dst]
                    t = dst_tok.sent.start
                    dst_cat = categ[dst]
                    if dst_cat == 'Matched':
                        curr_predicate = dst_tok
                        yield Frame(t, curr_predicate, ent, None)
                    elif dst_cat == 'Entity' and t == curr_t and curr_predicate:
                        dst_ent = ents[dst]
                        yield Frame(t, curr_predicate, ent, dst_ent)
                    curr_t = t

                elif edge_type == 'reverse' and categ[src] == 'Matched':
                    curr_predicate = doc[src]
                else: # edge_type == 'nontree' or ('reverse' but irrelevant)
                    curr_predicate = None
            
    def get_frames_as_df(self, NA=''):
        df = pd.DataFrame([
            (t, pred.lemma_, ag.root.text, pat.root.text if pat else NA)
            for t, pred, ag, pat in self.iter_frames()
        ])
        df = df.groupby(['t','predicate','agent','patient']).count().reset_index()
        return df

    def get_linear_layout(self):

        def fmt_node(tok, win=(3,3), clamp=(0, len(self.doc))):
            sent = tok.sent
            sent_toks = [t.text for t in sent]
            j = tok.i - sent.start
            wl, wr = win
            cl, cr = clamp
            j0 = j-wl if j > cl else j
            j1 = j+wr if j < cr else j
            sent_toks[j] = f"<b>{tok.text}</b>"
            return f"<b>{tok.text.upper()} ({tok.dep_})</b>: " + ' '.join(sent_toks[j0:j1])

        g = self.graph
        nodes = np.array(list(g.nodes))
        toks = list(doc_at(self.doc, nodes))
        ents = self.ents
        
        Xn = nodes
        Xe = np.array([[src, dst, None] for src, dst in g.edges]).flatten()
        levels = {n.i: (+20 if ents[n.i].start == n.i else +1)
                        if n.i in ents 
                        else -ilen(n.ancestors) for n in toks}
        Yn = np.array([levels[n] for n in nodes])
        Ye = np.array([[levels[src], levels[dst], None] for src, dst in g.edges]).flatten()
        Tn = np.array([fmt_node(n) for n in toks])
        Cn = pd.Series(self.categ, dtype="category").cat.codes
        
        return {
            'nodes': {
                'x': Xn,
                'y': Yn,
                'label': Tn,
                'color': Cn,
            },
            'edges': {
                'x': Xe,
                'y': Ye,
            }
        }

    def display_linear(self, title='Document Semantic Graph', show_labels=False, default_range=None):
        import plotly as py
        import plotly.graph_objs as go

        lyt = self.get_linear_layout()

        edge_trace = go.Scattergl(
            x=lyt['edges']['x'],
            y=lyt['edges']['y'],
            line=dict(color='rgba(125, 125, 125, 0.4)', width=.8),
            mode='lines' + ('+text' if show_labels else ''),
        )
        
        node_trace = go.Scattergl(
            x=lyt['nodes']['x'],
            y=lyt['nodes']['y'],
            text=lyt['nodes']['label'],
            mode='markers' + ('+text' if show_labels else ''),
            marker=dict(color=lyt['nodes']['color']),
            hoverinfo='text',
            textposition='top center',
        ) 
        
        layout = dict(
            title=title,
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
                range=default_range,
            ),
        )

        fig = go.Figure(data=[edge_trace,node_trace], layout=layout)

        py.offline.iplot(fig)