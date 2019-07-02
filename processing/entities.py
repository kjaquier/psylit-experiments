import pandas as pd
import numpy as np


def entity_classifier(tok_hypernyms_matcher):
    PERSON, ENV, UNK, NARR, READ, NAN = ['person', 'environment', 'unknown', 'narrator', 'reader', np.nan]
    
    exceptions = {
        ENV: {'it','this','that','its','itself','something'},
        UNK: {'they','them','themselves','their','these','those'},
        NARR: {'i',  'me' ,'myself',   'my',   'mine' },
        READ: {'you'      ,'yourself', 'your', 'yours'},
    }
    exceptions = {w:k for k, v in exceptions.items() for w in v}
    
    
    def classify_entity(cluster, mention):
        #is_proper | is_person_noun | is_person_unresolved | (ents.NER_PERSON > 0)
        c_root = cluster.main.root
        if c_root.pos_ == 'PROPN':  # resolved to an entity
            if c_root.ent_type_ == 'PERSON':
                return PERSON
            elif c_root.ent_type_ in {'NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT'}:
                return ENV
            else: # dates, quantities etc.
                return NAN # TODO cases where the mention is 'he' etc. so assign PERSON in those cases
        elif c_root.pos_ == 'NOUN':  # resolved to a noun
            hypernyms = {hmap[m.name()] for m in tok_hypernyms_matcher(c_root)}
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