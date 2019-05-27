import networkx

import flair

from flair.embeddings import FlairEmbeddings, BertEmbeddings

# init Flair embeddings
#flair_forward_embedding = FlairEmbeddings('multi-forward')
#flair_backward_embedding = FlairEmbeddings('multi-backward')

# init multilingual BERT
bert_embedding = BertEmbeddings('bert-base-cased')

from flair.embeddings import StackedEmbeddings

# now create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(
    embeddings=[
        #flair_forward_embedding,
        #flair_backward_embedding,
        bert_embedding])

import nltk
nltk.download('framenet_v17')

from nltk.corpus import framenet as fn
len(fn.frames())

txt=preprocess.read_pg(data_root + r'\EN_1818_Shelley,Mary_Frankenstein_Novel.txt')
print(len(txt), 'chars')

from segtok.segmenter import split_single
sentences = [Sentence(s, use_tokenizer=True) for s in split_single(txt)]
print(len(sentences), 'sentences')

import random as rand

t = range(100)#rand.sample(range(len(sentences)), 100)
sents_sample = [sentences[i] for i in sorted(t)]

t = np.array(t)
_ = bert_embedding.embed(sents_sample)

from scipy.spatial.distance import cosine
from torch.nn.functional import cosine_similarity
from itertools import product

def cosines(tokens):
    s = np.zeros([n,n])
    for (i, j), _ in np.ndenumerate(s):
        s[i, j] = cosine(tokens[i], tokens[j])
    return s

def cosines(vecs, return_type=np.zeros):
    vecs = list(vecs)
    n = len(vecs)
    c = return_type([n,n])
    # TODO compute all norms at once
    for i in range(n):
        vi = vecs[i]
        vi2 = th.norm(vi)
        for j in range(i, n):
            vj = vecs[j]
            vj2 = th.norm(vj)
            c[j, i] = c[i, j] = 1 - th.dot(vi, vj) / (vi2 * vj2)  #cosine(vecs[i], vecs[j])
    return c

toks = [t for s in sents_sample for t in s]
toks_str = np.array([t.text for t in toks])

D = cosines(t.embedding.numpy() for t in toks)

def show_similarities(tokens, beta=0.1):
    D = cosines(t.embedding.numpy() for t in tokens)
    S = np.exp(-beta * D / D.std())
    
    fig = plt.figure()
    ax = plt.gca()
    im = ax.matshow(S,aspect='auto')
    fig.colorbar(im)
    ax.set_xticklabels(toks_str)
    ax.set_yticklabels(toks_str)
    # Set ticks on both sides of axes on
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # Rotate and align bottom ticklabels
    plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
             ha="right", va="center", rotation_mode="anchor")
    ax.set_title("Token similarity (BERT cosine)")
    #fig.tight_layout()
    plt.show()
    
    dist = np.triu(S).flatten()
    dist_plus = dist[dist > 0]
    print('fraction > 0:',len(dist_plus) / len(dist))
    plt.hist(dist_plus, bins=300)
    plt.title('S distrib')
    plt.show()
    
    mean = np.mean(dist)
    std = np.std(dist)
    print('mean:', mean)
    print('1sigma:', mean + 1*std)
    print('2sigma:', mean + 2*std)

beta = 0.1
S = np.exp(-beta * D / D.std())

fig = plt.figure()
ax = plt.gca()
im = ax.matshow(S,aspect='auto')
fig.colorbar(im)
ax.set_xticklabels(toks_str)
ax.set_yticklabels(toks_str)
# Set ticks on both sides of axes on
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
# Rotate and align bottom ticklabels
plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="right", va="center", rotation_mode="anchor")
ax.set_title("Token similarity (BERT cosine)")
#fig.tight_layout()
plt.show()

dist = np.triu(S).flatten()
dist_plus = dist[dist > 0]
print('fraction > 0:',len(dist_plus) / len(dist))
plt.hist(dist_plus, bins=300)
plt.title('S distrib')
plt.show()

thres = np.mean(dist) + 1*np.std(dist)
thres

STOPS = set(t.text for t in nlp.vocab if t.is_stop)

S_flat = np.triu(S - np.eye(S.shape[0])).flatten()
highest_i_flat = np.argwhere(S_flat > thres).flatten()
highest_i = np.array(np.unravel_index(highest_i_flat, S.shape))
highest_i

S_highest = S[highest_i[0,:],highest_i[1,:]]

most_similars_str = toks_str[highest_i]
df = pd.DataFrame({'t0':most_similars_str[0,:], 't1':most_similars_str[1,:], 'similarity':S_highest})
df = df[~df.t0.isin(STOPS) & ~df.t1.isin(STOPS) ]
df

ner_like = df.t0[~df.t0.isin(STOPS)
                 & ~df.t0.str.isupper() 
                 & df.t0.str.get(0).str.isupper()
                ].value_counts()
ner_like[ner_like > 1]

alltoks = set(toks_str)
alltoks

def querydf(df, included=None, excluded=None):
    included = included if included is not None else ~df.t0.isnull()
    excluded = excluded if excluded is not None else df.t0.isnull()
    d = df[(df.t0 != df.t1) & included & ~excluded]
    return d.groupby(['t0','t1']).mean().sort_values('similarity', ascending=False)

querydf(df,
       included = (df.t0 == 'me') & df.t1.str.get(0).str.islower(),
       )
        #excluded = df.t1.isin({'Miss','Mrs.','ladies','lady','girl','girls','her'}))

lady_bow = {'Miss','Mrs.','ladies','lady','girl','girls','her'}
included = df.t1.isin(lady_bow) & df.t0.str.get(0).str.isupper()
excluded = df.t0.isin(lady_bow)
df[(df.t0 != df.t1) & included & ~excluded].groupby(['t0','t1']).mean().sort_values('similarity', ascending=False)

included = (df.t0 == 'Emma') & df.t1.str.get(0).str.islower()
excluded = df.t1.isin({'Miss','Mrs.','ladies','lady','girl','girls','her'})
emma = df[(df.t0 != df.t1) & included & ~excluded]
emma.groupby(['t0','t1']).mean().sort_values('similarity', ascending=False)

df[(df.t0 == 'Agatha') & (df.t1 == 'Emma')].plot()

tdf = pd.DataFrame({'tok':toks_str})

counts = tdf.tok[~tdf.tok.isin(STOPS) & tdf.tok.str.get(0).str.isupper()].value_counts()
counts[counts > 1]
#tdf[tdf.tok == 'mourning']

def trajectories(toks, ref_tok, traj_toks):
    toks_str = np.array([t.text for t in toks])
    ref_vecs = [t.embedding for t in toks if t.text == ref_tok]
    print(len(ref_vecs), 'vecs for', ref_tok)
    ref_vec = th.mean(th.stack(ref_vecs), dim=0)
    trajs = []
    for tok in traj_toks:
        vecs = [t.embedding for t in toks if t.text == tok]
        print(len(vecs), 'vecs for', tok)
        #diffs = np.array([(v - ref_vec).numpy() for v in vecs])
        #sims = np.linalg.norm(diffs, axis=1)
        sims = np.array([cosine(v.numpy(), ref_vec.numpy()) for v in vecs])
        trajs.append(sims)
    return trajs

def plot_trajs(toks, ref_tok, traj_toks):
    trajs = trajectories(toks, ref_tok, traj_toks)
    for traj in trajs:
        plt.plot(traj)
    plt.title(f"Distance to '{ref_tok}'")
    plt.legend(traj_toks)

plot_trajs(toks, 'enthusiasm', ['me','you',''])
plt.show()

df[(df.t1 == 'mourning') & (df.t0.isin(['Agatha','Tittens']))]


toks_unique = np.unique(toks_str)
toks_selected = np.unique(toks_str[highest_i][0,:])
m = len(toks_selected)
e = [(toks_selected[i],toks_selected[j]) for i in range(m) for j in range(i,m)]

import networkx as nx
from networkx.algorithms import community as nxcom

G = nx.Graph()
G.add_nodes_from(toks_unique)
G.add_edges_from(e)
print(nx.number_of_nodes(G), 'nodes')
print(nx.number_of_edges(G), 'edges')

#nx.get_edge_attributes(G,'weight')
comms = list(nxcom.asyn_lpa_communities(G))#, weight='weight'))#, weight='weight'))


set(len(c) for c in comms)

import networkx.drawing.nx_pylab as nxd
nxd.draw_circular(G)

len(toks), len(toks[0].embedding)

X = np.array([t.embedding.numpy() for t in toks])
X.shape

Xdf = pd.DataFrame(X)

Xcorr = Xdf.corr()


fig = plt.figure()
ax = plt.gca()
im = ax.matshow(Xcorr, aspect='auto')
fig.colorbar(im)
plt.show()

from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD

pca = PCA()
pca.fit(X)
#svd = TruncatedSVD(n_components=100)
#svd.fit(X)


#%%
plt.plot(pca.explained_variance_ratio_)#np.log(s**2/N))
plt.title("Variance per eigenvectors")
plt.xlabel("Eigenvector")
plt.ylabel("Variance")
plt.show()

#%%
plt.plot(np.cumsum(pca.explained_variance_ratio_))#np.log(s**2/N))
plt.title("Cumulative variance per eigenvectors")
plt.xlabel("Eigenvector")
plt.ylabel("Variance")
plt.ylim([0,1])
plt.show()


STOPS_EXTENDED = {t.upper() for t in 
                  STOPS.union({'most','as','of',',','.',';','the','thus','(',')','no','and','on','by','this','in','which','or','shall','to','a',':'})}

import sklearn as sk
import sklearn.cluster as skc
from math import sqrt

# Cluster analysis of PCA-reduced embeddings

comps = pca.components_[:1500,:]
clust = skc.AgglomerativeClustering(n_clusters=50)
lbls = clust.fit(comps).labels_
counts = np.bincount(lbls[lbls >= 0])
counts, len(lbls[lbls < 0])

for clust_id, clust_cnt in enumerate(counts):
    print('-'*5, f"cluster #{clust_id} ({clust_cnt} items)")
    clust_comps = comps[lbls == clust_id,:]
    loadings = np.sum(clust_comps, axis=1)
    features_idx_sorted = np.argsort(-loadings)[:20]
    terms = toks_str[features_idx_sorted]
    terms_loadings = loadings[features_idx_sorted]
    print('\n'.join(f"{t:16} {l:4.4g}" for t, l in zip(terms, terms_loadings) if t.strip().upper() not in STOPS_EXTENDED))
    print()

# Cluster analysis of pairwise similarities

clust = skc.AgglomerativeClustering(n_clusters=10)
lbls = clust.fit(S).labels_
counts = np.bincount(lbls[lbls >= 0])
counts, len(lbls[lbls < 0])

for clust_id, clust_cnt in enumerate(counts):
    print('-'*5, f"cluster #{clust_id} ({clust_cnt} items)")
    clust_comps = S[lbls == clust_id,:]
    loadings = np.sum(clust_comps, axis=1)
    loadings = loadings / loadings.sum()
    features_idx_sorted = np.argsort(-loadings)[:50]
    terms = toks_str[features_idx_sorted]
    terms_loadings = loadings[features_idx_sorted]
    print('\n'.join(f"{t:16} {l:4.4g}" for t, l in zip(terms, terms_loadings) if t.strip().upper() not in STOPS_EXTENDED))
    print()

from itertools import product

def compare_sentences(ss):
    tss = [[Sentence(sen, use_tokenizer=True) for sen in split_single(s)]
           for s in ss] # List[List[Sentence]]

    vs = [bert_embedding.embed(ts) for ts in tss] 

    t = 0
    C = []
    xstr = []
    ystr = [str((0, i)) for i in range(1, len(ss))]
    for sss in zip(*tss):
        for ts in zip(*sss):
            t1, *tothers = ts
            tstr = f"[{t1.text}] {' / '.join(t.text for t in tothers)}"
            cs = []
            for i, t2 in enumerate(tothers, 1):
                c = cosine(t1.embedding.numpy(), t2.embedding.numpy())
                cs.append(c)
                #xstrs.append(f'{t1.text}' / '{t2.text}')
            C.append(cs)
            xstr.append(tstr)
            t += 1
    C = np.array(C)

    Cs = np.exp(-0.1 * C / C.std())

    fig = plt.figure()
    ax = plt.gca()
    for i in range(C.shape[1]):
        ax.plot(Cs[:,i])
    ax.legend(ss[1:])
    plt.xticks(range(len(xstr)), xstr)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
             ha="right", va="center", rotation_mode="anchor")
    ax.set_title(f"Similarity with '{ss[0]}'")
    plt.ylim([0,1])
    plt.grid(True)

compare_sentences([
    'This product is very good.',
    'This product is very decent.',
    'This product is not good.',
    'This product is very bad.',
    'This product is not bad.',
    'This product is very thin.',
])
plt.show()

compare_sentences([
    'Alice did cry. An hour later, after they left, Alice was cooking. Then he came back.',
    'Alice did cry. An hour later, after they left, she was cooking. Then he came back.',
    'Alice did cry. An hour later, after they left, Emma was cooking. Then he came back.',
    'Alice did not cry. Hours later, after they left, Alice was cooking. Then he came back.',
    'Alice did not cry. Hours later, after they left, she was cooking. Then he came back.',
    'Alice did not cry. Hours later, after they left, Emma was cooking. Then he came back.',
])
plt.show()

import random as rand 

def rand_sample(n, *dims):
    for i in range(n):
        sample = []
        for d in dims:
            sample.append(rand.choice(d))
        yield sample

ents = ['Alice', 'Emma']#, 'he', 'she', 'they']
verbs = ['cry', 'laugh','shout','cook','fish','run','walk']
preps = ['back']#, 'in', 'out', 'over']
negs = ['did {}. An hour']#, 'did not {}. Hours']
time = ['later']#, 'earlier']

compare_sentences([

    f"{e1} {n.format(v1)} {time}, after {e2} left, {e3} was {v2}ing. Then {e4} came {p}"
    for e1, e2, e3, e4, v1, v2, n, time, p in rand_sample(30, ents, ents, ents, ents, verbs, verbs, negs, time, preps)
])
plt.legend([])
plt.show()


from sys import getsizeof
locs = pd.DataFrame({(k, getsizeof(v)) for k,v in locals().items()})

locs.sort_values(by=1,ascending=False)

del most_similars_i

%reset out