#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'psylit-experiments'))
	print(os.getcwd())
except:
	pass

#%%

#%load_ext autoreload
#%autoreload 2

from collections import Counter, defaultdict
from os import linesep as EOL
import re

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
plt.rcParams['figure.figsize'] = 16,10
np.random.seed(0)

from tic import preprocess

#%%
import spacy
nlp = spacy.load('en_core_web_sm')   # 'Vanilla' spacy model: spacy.load('en_core_web_sm')

#%%
import neuralcoref

coref = neuralcoref.NeuralCoref(nlp.vocab)
merge_ents = nlp.create_pipe("merge_entities")

nlp.add_pipe(coref, name='neuralcoref')
nlp.add_pipe(merge_ents, after="ner")

#%%
#filename = r'data\SentiWordNet_3.0.0.txt'
#df = pd.read_csv(filename, sep='\t', delimiter=None, header='infer', comment='#')

#%%
whole_txt = preprocess.read_pg(r'data\pg9296_clarissa_harlowe_vol1.txt')

# #%%
# from allennlp.predictors.predictor import Predictor

# #%%

# class CorefResolver():

#     def __init__(self, label='REFERENCE'):
#         self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
#         self.label = label


#     def __call__(self, doc):
#         self.predictor.predict(doc.text)
#         clusters = self.predictor['clusters']
#         for clust_id, cluster_spans in enumerate(clusters):
#             cluster_name = cluster_spans[0]
            
#             for start, end in cluster_spans:
#                 span = spacy.Span(doc, start, end, label=self.label)
#                 print(span)
#             print("---")
#         return doc

# nlp.add_pipe(CorefResolver(), after='tagger')

#%%

doc = nlp(whole_txt[:20000])
# doc.to_disk('docs/harlowe.spdoc')

#%%
df = pd.DataFrame([
    (tok.i, tok.text, tok.pos_, tok.dep_, tok.lemma_, tok._.in_coref, 
     tok._.coref_clusters[0].main.text if tok._.in_coref else None,
     tok.cluster, tok.sentiment, 
     tok.is_stop, tok.is_oov, tok.is_space,
     len(list(tok.ancestors)))
    for tok in doc
])
df.columns = [
     "i",
     "text",
     "pos",
     "dep",
     "lemma",
     "in_coref",
     "coref",
     "cluster",
     "sentiment",
     "is_stop",
     "is_oov",
     "is_space",
     "n_ancestors"]

df = df[~df.is_space]
del df['is_space']


#%%

counts = df.groupby(['pos','in_coref'])['i'].count()
counts = counts.unstack(level=1)
counts = counts.fillna(0)
#counts.columns = counts.columns.droplevel(level=0)
counts['Total'] = counts[False] + counts[True]
counts

#%%
counts = (df.groupby(['pos','in_coref'])
            .size()
            .sort_values()
            .reset_index()
            .pivot(columns='pos', index='in_coref')
            .fillna(0))
            #.reset_index())
counts
#counts.plot(x='pos', kind='bar', stacked=True)

#%%
resolved_objs = (df[df.dep.isin(['nsubj','nsubjpass','dobj','pobj','iobj'])]
                  .coref
                  .value_counts(dropna=False))
found_entities = pd.Series([e.text for e in doc.ents]).value_counts()

#%%
resolved_objs

#%%
found_entities

#%%
ss = list(doc.sents)[213:218]

#%%

def find_related_entities(token):
    if not token._.in_coref:
        return

    clusters = list(token._.coref_clusters)
    while clusters:
        c = clusters.pop()

        s = c.main
        if s._.is_coref:
            clusters.append(s._.coref_cluster)
        else:
            yield s

for e in find_related_entities(ss[0][0]):
    print("Found:", e)

#%%
for sent in list(doc.sents)[213:223]:
    print("'''"+sent.text+"'''")
    print([c.main for t in sent for c in t._.coref_clusters])
    print()


#%%
