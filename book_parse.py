#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(r'C:\Users\kevin\Documents\Workspace\psylit-experiments')
	print(os.getcwd())
except Exception as e:
	print(e)
#%%

import pandas as pd
import numpy as np

import tools
import tools.spacy as myspacy
import tools.pandas as mypd

import processing
import processing.lexicons as lexicons
import processing.preprocess as preprocess

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 
import neuralcoref

#%%

data_root = r'..\datasets\2_txtalb_Novel450'
book_filename = r'\EN_1818_Shelley,Mary_Frankenstein_Novel.txt'



#%%

def build_pipe(model='en_core_web_sm'):

	nlp = spacy.load(model)

	merge_ents = nlp.create_pipe("merge_entities")
	nlp.add_pipe(merge_ents, after="ner")

	#nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')

	coref = neuralcoref.NeuralCoref(nlp.vocab, max_dist=20, store_scores=False)
	nlp.add_pipe(coref, name='neuralcoref')

	nrc_lex = lexicons.load_nrc_wordlevel()
	lextag = myspacy.LexiconTagger(nlp.vocab, nrc_lex)
	nlp.add_pipe(lextag)

	negtag = myspacy.NegTagger(nlp.vocab)
	nlp.add_pipe(negtag)

	semdep = myspacy.SemanticDepParser()
	nlp.add_pipe(semdep)

	predparser = myspacy.PredicateParser(nlp.vocab)
	nlp.add_pipe(predparser)

	return nlp

#

#%%
#

#%%

def get_dataframe(doc):
	fmt = lambda tok: tok.text.strip().lower()[:50] if tok else None
	data = [
		{'i': tok.i,
		't': tok.sent.start,
		'neg': tok._.negated,
		'lemma': tok.lemma_[:50],
		'text': fmt(tok),
		**{('R_'+r): fmt(t) for r, t in tok._.sem_deps},
		**{('L_'+doc.vocab[cat].text): 1.0 for cat in tok._.lex},
		}
		for tok in doc if tok._.has_lex
	]

	table = pd.DataFrame(data)
	rel_cols = table.columns[list(table.columns.str.startswith('R_'))]
	lex_cols = table.columns[list(table.columns.str.startswith('L_'))]
	table[lex_cols] = table[lex_cols].fillna(0)
	return table


#%%

def main(input_filename:"Raw text of book to read (UTF-8)",
		 output_filename:"CSV file to write",
		 save_doc:("File where spacy Doc object is saved",'option','d')=None,
		 start:("Position to read from",'option','s')=None,
		 end:("Position to stop reading after",'option','e')=None):

	txt = preprocess.read_pg(input_filename)
	print('start=', repr(start), ' end=', repr(end))
	start = int(start) if start is not None else 0
	end = int(end) if end is not None else None
	n = len(txt)

	txt = txt[start:] if end is None else txt[start:end]
	
	print('Processing', len(txt), '/', n, 'chars')

	nlp = build_pipe()
	print("Pipeline: ", ', '.join(pname for pname, _ in nlp.pipeline))
	doc = nlp(txt)

	if save_doc:
		doc.to_disk(save_doc)

	df = get_dataframe(doc)
	print("Data frame size: ", df.size)
	df.to_csv(output_filename)


if __name__ == '__main__':
	import plac
	plac.call(main)