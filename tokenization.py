from os import linesep as EOL
import re

import pandas as pd
#from matplotlib import pyplot as plt
import numpy as np
#import seaborn as sns

from gutenberg.cleanup import strip_headers
#import flair
import spacy
from spacy.symbols import nsubj, VERB, ADJ, root

has_info = lambda t: t.dep_ in {'acomp','pobj','neg','iobj','obj','nounmod','nsubjpass'}
is_entity = lambda t: t.dep == nsubj or t.ent_iob == 'I' or t.lemma_ == '-PRON-'
is_interesting = lambda t: t.dep_ and (has_info(t) or is_entity(t))

def token_fmt(t):
	if t.ent_iob == 'I':
		return f"<{t.lemma_}>"
	if t.dep == nsubj:
		return f"^{t.lemma_}^"
	# if t.dep == root:
	# 	return f"~{t.lemma_}~"
	else:
		return f"'{t.lemma_}'"

def pred_and(*preds):
	def pred_cmp(x):
		for p in preds:
			if not p(x):
				return False
		return True
	return pred_cmp

def to_tree(t):
	lefts = [to_tree(n) for n in t.lefts]
	rights = [to_tree(n) for n in t.rights]
	return (lefts, t, rights)

def prune_nodes(nodes):
	for lefts, token, rights in nodes:
		lefts = [prune(subtree) for subtree in lefts]
		rights = [prune(subtree) for subtree in rights]
				
		if token.text.strip() and is_interesting(token):
			yield (lefts, token, rights)
		
		else:
			yield from lefts
			yield from rights
		
def prune(tree):
	lefts, token, rights = tree

	lefts = list(prune_nodes(lefts))
	rights = list(prune_nodes(rights))
	return (lefts, token, rights)

def print_tree(tree, prefix=""):
	lefts, token, rights = tree
	print(prefix, token_fmt(token), f"[{token.dep_}]")
	for subtree in lefts:
		print_tree(subtree, prefix + "  <")
	for subtree in rights:
		print_tree(subtree, prefix + "  >")
	

def match_parents(*y):
	return lambda token: (t for t in token.ancestors if t.dep_ in y)

def tikenize(doc):	
	roots = (t for t in doc if t.head == t)

	for root in roots:
		tree = to_tree(root)
		yield prune(tree)
		# for token in enumerate(root.subtree):
			
		# 	ancs = list(y.ancestors)
		# 	lvl = len(ancs)
		# 	print("-- " * lvl, f"{repr(y.lemma_)} [{y.dep_}]", ancs)

		# for token in doc:
		# 	if token.dep != root:
		# 		if is_marker(token):
		# 			yield tuple(a for a in token.ancestors if is_marker(a) or is_entity(a) or a.dep == root) + (token,)
		
def tikenize_old(doc):
	# for i, ent in enumerate(doc.ents):
	# 	break
	# 	if ent.label_ not in ('PERSON','NORP','GPE','LOC','PRODUCT','EVENT','LANGUAGE','ORG'):
	# 		continue
			
	# 	ent_name = ' '.join(ent.text.strip().split()).lower()
	# 	yield ent_name

	for token in doc:
		if token.dep == nsubj:# and token.text.upper() in ent_counter.keys():
			if token.head.pos == VERB:
				#pos_lbl = sym_lbl[token.head.pos]
				#print(f"{token.lemma_:12} {token.head.lemma_:10}({pos_lbl})")
				yield token, token.head
		if token.pos == ADJ:# and token.head.text.upper() in ent_counter.keys():
			#pos_lbl = sym_lbl[token.pos]
			#print(f"{possible_subject.head.lemma_:12} {possible_subject.lemma_:10}({pos_lbl})")
			yield token.head, token

