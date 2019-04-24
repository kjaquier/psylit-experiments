#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os

try:
    os.chdir(os.path.join(os.getcwd(), 'psylit-experiments'))
except:
    print("failed to change dir")
print(os.getcwd())


#%%
from collections import Counter, defaultdict
from os import linesep as EOL
from itertools import islice

from tic import cleanup
from tic import preprocess
from tic import utils

#%%
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 

nlp = spacy.load('en_core_web_sm')
merge_ents = nlp.create_pipe("merge_entities")

nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
nlp.add_pipe(merge_ents, after="ner")

#%%
filename = r'data\pg9296_clarissa_harlowe_vol1.txt'
whole_txt = preprocess.read_pg(filename)

prep = preprocess.BasicPreprocessor(nlp)
tokens = prep(whole_txt)
prep.stats()

#%%
pe = preprocess.PsychFeatsExtractor()
ptoks = list(pe(tokens))
pe.stats()

#%%
utils.write_tokens(ptoks, r'outputs\ch1_affect.test.tsv', sep=r'\t')
