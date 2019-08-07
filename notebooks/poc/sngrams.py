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

merge_ents = nlp.create_pipe("merge_entities")

nlp.add_pipe(merge_ents, after="ner")


#%%
whole_txt = preprocess.read_pg(r'..\datasets\EN_1889_Doyle,ArthurConan_TheMysteryoftheCloomber_Novel.txt')


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
