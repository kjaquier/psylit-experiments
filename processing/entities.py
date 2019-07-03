from itertools import combinations

from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np


def tok_hypernyms_matcher(targets, highest_level=0, highest_common_level=0):
    targets = {wn.synset(k) for k in targets}

    def matcher(token):
        matched = set()
        
        hypers_all = hypers_common = {h for s in token._.wordnet.synsets() for h in s.hypernyms()}
        matched |= targets & hypers_all
        for _ in range(highest_level):
            hypers_all = {h for s in hypers_all for h in s.hypernyms()}
            if not hypers_all:
                break
            matched |= targets & hypers_all

        for _ in range(highest_common_level):
            hypers_pairs = combinations(hypers_common, 2)
            hypers_common = {h for h1, h2 in hypers_pairs for h in h1.lowest_common_hypernyms(h2)}
            if not hypers_common:
                break
            matched |= targets & hypers_common

        return matched
    
    return matcher


def entity_classifier():
    PERSON, ENV, UNK, NARR, READ, NAN = ['person', 'environment', 'unknown', 'narrator', 'reader', np.nan]
    
    exceptions = {
        ENV: {'it','this','that','its','itself','something'},
        UNK: {'they','them','themselves','their','these','those'},
        NARR: {'i',  'me' ,'myself',   'my',   'mine' },
        READ: {'you'      ,'yourself', 'your', 'yours'},
    }
    exceptions = {w:k for k, v in exceptions.items() for w in v}
    
    hypernym_map = {'person.n.01': 'person', 
                    'female.n.02':'female', 
                    'woman.n.01':'female',
                    'man.n.01':'male',
                    'male.n.02': 'male', 
                    'entity.n.01': 'entity'}

    hmatcher = tok_hypernyms_matcher(hypernym_map.keys(),
                                     highest_level=10,
                                     highest_common_level=0)

    
    def classify_entity(cluster, mention):
        c_root = cluster.main.root
        if c_root.pos_ == 'PROPN':  # resolved to an entity
            if c_root.ent_type_ == 'PERSON':
                return PERSON
            elif c_root.ent_type_ in {'NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT'}:
                return ENV
            else: # dates, quantities etc.
                return NAN # TODO cases where the mention is 'he' etc. so assign PERSON in those cases
        elif c_root.pos_ == 'NOUN':  # resolved to a noun
            hypernyms = {hypernym_map[m.name()] for m in hmatcher(c_root)}
            if 'person' in hypernyms:
                return PERSON
            else:
                return ENV
        elif c_root.pos_ in {'DET', 'PRON'}: # unresolved
            c_text = c_root.text.lower()
            m_text = mention.root.text.lower()
            
            ret = exceptions.get(c_text, None)
            ret = ret or exceptions.get(m_text, None)
            
            # unlikely a det/pron doesn't refer to a person (except in cases above)
            return ret or PERSON

        else: # verbs, adj etc... most likely irrelevant (or unknown?)
            return UNK
        
    return classify_entity