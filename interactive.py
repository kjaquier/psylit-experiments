#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'psylit-experiments'))
	print(os.getcwd())
except:
	pass

#%%

%load_ext autoreload
%autoreload 2

from collections import *
from os import linesep as EOL
import re

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#import seaborn as sns
#plt.rcParams['figure.figsize'] = 16,10
np.random.seed(0)

from gutenberg.cleanup import strip_headers
#import flair
import spacy
nlp = spacy.load('en')

from tokenization import *

#%%
filename = r'..\tic-personality-words\resources\Text Files\08 David Copperfield text.txt'
filename = r'pg4693_FamousAffinitiesOfHistory.txt'
with open(filename, encoding='utf8') as f:
    lines = [l.strip() for l in f.readlines()]
    whole_txt = EOL.join(l for l in lines if l)
    whole_txt = strip_headers(whole_txt)

#%%

def clean_title(t):
    t = re.sub('THE STORY OF', '', t)
    t = re.sub('THE MYSTERY OF', '', t)
    return t.strip()
    

stories = whole_txt.split(EOL)[5:39]
subjects = EOL.join(clean_title(s) for s in stories)
subjects = set(e.lemma_.strip() for e in nlp(subjects).ents)
subjects -= {''}
subjects

#%%
doc = nlp(whole_txt)

#%%
testtxt = """
	Alice said that James Bond was attractive, but James was more attracted to computers. He did say Alice was not nice enough.
	Oscar likes Alice though, but Alice doesn't like him. Although he has many friends, James Bond isn't one of them.
	James Bond is as sick as a drunk man.
    George Washington comes to Washington whenever George Do goes to Do.
"""
testdoc = nlp(testtxt)

#%%
for tree in tikenize(testdoc):
	print_tree(tree)
	print("-"*50)
	#print(" ".join(t.lemma_ for t in tiken))

#%%

sym_lbl = {VERB: "VERB", ADJ: "ADJ"}
counters = defaultdict(Counter)
for tiken in tikenize_old(doc):
    entity, *infos = tiken
    if entity.ent_iob_ not in ("B","I"):
        continue
    counters[entity.lemma_.strip()][infos[0].lemma_.strip()] += 1

totals = {ent: sum(counters[ent].values()) for ent in counters.keys()}

print("counting done")

#%%
def write_results(counters, filename, max_subj=None, max_info=None):
    subjs = sorted(totals.items(), key=lambda x: -x[1])
    if max_subj:
        subjs = subjs[:max_subj]
    with open(filename, "w") as fw:
        for ent, ent_count in subjs:
            infos = counters[ent].most_common()
            if max_info:
                infos = infos[:max_info]
            fw.write(f"{ent.upper()} ({ent_count}):\n")
            fw.write(" "*16)
            fw.write(", ".join(f"'{i}' ({c})" for i, c in infos) + "\n"*2)

#%%
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load('ner-fast')
import syntok
from segtok.segmenter import split_single


#%%
sents = [Sentence(sent, use_tokenizer=True) for sent in split_single(testtxt)]


for i, sent in enumerate(sents):
    tagger.predict(sent)
    print(i, "-"*20)
    #print(sent.to_tagged_string())
    for x in sent.get_spans('ner'):
        print(">",x.text,x.tag)
    
    
#%%

def tokenization_by_noun_chunks(doc):
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ == 'nsubj':
            subj = chunk.root.text.upper()
            info = chunk.root.head.lemma_
        elif chunk.root.dep_ == 'dobj':
            subj = chunk.root.text.upper()
            info = chunk.root.head.lemma_+">"
        elif chunk.root.dep_ == 'pobj':
            subj = chunk.root.text.upper()
            info = " ".join(a.text for a in list(chunk.root.ancestors)[:2][::-1])+">"
        elif chunk.root.dep_ == 'conj':
            subj = chunk.root.text.strip().upper() + f"[{chunk.root.dep_}]"
            #ances = "|".join(t.text for t in list(chunk.root.ancestors)[::-1]))
            ances = [t.lemma_ for t in chunk.root.ancestors 
                    if t.pos_ in ('VERB', 'ADP', 'PART')][::-1]
            if ances:
                info = " ".join(ances) + ">"
            else:
                continue
        elif chunk.root.dep_ == 'appos':
            continue
            subj = chunk.root.text.upper() + f"[{chunk.root.dep_}]"
            ances = [t.lemma_ for t in chunk.root.ancestors 
                    if t.pos_ in ('VERB', 'NOUN', 'ADP', 'PART')][::-1]
            descs = [t.lemma_ for t in chunk.root.subtree 
                    if t.pos_ in ('VERB', 'NOUN', 'ADP', 'PART')][::-1]
            cnt[subj][" ".join(ances) + ">"] += 1
            cnt[subj][" ".join(descs)] += 1
            continue
        else:
            continue
            subj = f'-NOT IMPL: {chunk.root.dep_}-'
            ances = list(chunk.root.ancestors)
            if ances:
                sent_root = ances[-1]
                whole_sent_txt = " ".join(t.text for t in sent_root.subtree).strip()
                info = f"#{chunk.root.text}# {whole_sent_txt}"
            else:
                info = chunk.root.text
        yield subj, info


#%%
cnt = defaultdict(Counter)
for subj, info in tokenization_by_noun_chunks(doc):
    cnt[subj][info] += 1

totals = {ent: sum(cnt[ent].values()) for ent in cnt.keys()}

print("counting done")

#%%

write_results(cnt,"counts2.txt")

#%% 
def write_tokens(tokens, filename, sep=';', linesep=EOL):
    with open(filename, 'w') as fw:
        i = 0
        for toks in tokens:
            fw.write(sep.join(toks) + linesep)
            i += 1
        print(i, "events extracted")

toks = tokenization_by_noun_chunks(doc)
write_tokens(toks, "tokens.txt")
print("done")
#%%

# for ent, ent_count in sorted(totals.items(), key=lambda x: -x[1]):
#     print(f"{ent.upper()} ({ent_count}):")
#     print(" "*16,end="")
#     print(", ".join(f"'{i}' ({c})" for i, c in cnt[ent].most_common()[:1000]))

# with open("counts2.txt", "w") as fw:
#     for ent, ent_count in sorted(totals.items(), key=lambda x: -x[1])[:100]:
#         fw.write(f"{ent.upper()} ({ent_count}):\n")
#         fw.write(" "*16)
#         fw.write(", ".join(f"'{i}' ({c})" for i, c in cnt[ent].most_common()[:1000]) + "\n"*2)

#%%
from functools import reduce

def coocc_matrix(counters, n_entities, n_words):

    counter_sum = lambda c1, c2: c1 + c2
    trait_cnt = reduce(lambda c1, c2: c1+c2, counters.values())

    trait_id = {trait: i for i, trait in enumerate(trait_cnt.keys())}
    most_common_ents = sorted(totals.items(), key=lambda x: -x[1])[:n_entities]
    X = np.array([
        [cnt[ent][k]/total for k, total in trait_cnt.most_common()[:n_words]]
        for ent, _ in most_common_ents
    ])

    return X

X = coocc_matrix(cnt, 200, 1000)

#%%
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
#pca = PCA()
#pca.fit(X)
svd = TruncatedSVD(n_components=100)
svd.fit(X)


#%%
plt.plot(svd.explained_variance_ratio_)#np.log(s**2/N))
plt.title("Variance per eigenvectors")
plt.xlabel("Eigenvector")
plt.ylabel("Variance")
plt.show()

#%%
plt.plot(np.cumsum(svd.explained_variance_ratio_))#np.log(s**2/N))
plt.title("Cumulative variance per eigenvectors")
plt.xlabel("Eigenvector")
plt.ylabel("Variance")
plt.ylim([0,1])
plt.show()


#%%
import pickle
with open("pca.pickle", 'wb') as fw:
    pickle.dump(pca.explained_variance_, fw, protocol=0) # protocol 0 is printable ASCII

#%%
