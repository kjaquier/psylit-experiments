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
#from matplotlib import pyplot as plt
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
with open('pg4693_FamousAffinitiesOfHistory.txt') as f:
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
	Alice said that James Bond was attractive. 
	But James Bond was more attracted to computers. He said Alice was not nice enough.
	Oscar likes Alice though, but Alice doesn't like Oscar. 
	Oscar has many friends though James Bond isn't one of them.
	James Bond is as sick as a drunk man.
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
with open("counts.txt", "w") as fw:
    for ent, ent_count in sorted(totals.items(), key=lambda x: -x[1])[:100]:
        fw.write(f"{ent.upper()} ({ent_count}):\n")
        fw.write(" "*16)
        fw.write(", ".join(f"'{i}' ({c})" for i, c in counters[ent].most_common()[:1000]) + "\n"*2)

#%%
