from itertools import combinations

from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np

from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token


class RemoveExtensionsMixin: # TODO import or refactor
    """DUPLICATE OF tools.spacy.RemoveExtensionsMixin"""

    def __init__(self, extensions=None, **kwargs):
        self.exts = []
        self.kwargs = kwargs
        for cls, attr_name, kw in (extensions or []):
            self.set_extension(cls, attr_name, **kwargs, **kw)

    def set_extension(self, cls, attr_name, **kwargs):
        print(f"[{self.__class__.__name__}] set extension {cls.__class__.__name__}._.{attr_name} {kwargs!r}")
        cls.set_extension(attr_name, **self.kwargs, **kwargs)
        self.exts.append((cls, attr_name, kwargs))

    def remove_extensions(self):
        for cls, attr_name, _ in self.exts:
            print(f"[{self.__class__.__name__}] remove extension {cls.__class__.__name__}._.{attr_name}")
            cls.remove_extension(attr_name)

    def get_extensions_remover_component(self):
        
        def component(doc):
            self.remove_extensions()
            return doc

        return component


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


class EntityClassifier(RemoveExtensionsMixin):

    name = 'ent_class'

    def __init__(self, span_attr='ent_class', doc_attr='classified_ents', force_ext=False):
        print("[UNUSED] use entity_classifier")
        super().__init__(force=force_ext)
        super().set_extension(Span, span_attr, default=None)
        super().set_extension(Doc, doc_attr, default=None)
        self.span_attr = span_attr
        self.doc_attr = doc_attr

    def __call__(self, doc):
        return doc


def entity_classifier(vocab, doc_attr='classified_ents', force_ext=False):
    PERSON, ENV, UNK, NARR, READ, NAN = ['person', 'environment', 'unknown', 'narrator', 'reader', np.nan]
    
    exceptions = {
        ENV:     {'it','this','that','its','itself','something'},
        UNK:   {'they','them','themselves','their','these','those'},
        NARR:    {'i',  'me' ,'myself',   'my',   'mine' },
        READ:    {'you'      ,'yourself', 'your', 'yours'},
        PERSON:  {'he', 'him','himself', 'his', 
                    'she','her','herself',          'hers'},
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

    
    exceptions_matcher = Matcher(vocab)
    exceptions_matcher.add('detpron', None, [{'POS': {'IN': ('DET','PRON')}}])#, '_': {'in_coref': False}}])

    def iter_entities(doc):
        import time
        t = time.clock()
        coref_tokens = {
            t.i for c in doc._.coref_clusters for m in c.mentions for t in m
        }#doc._.coref_tokens
        t = time.clock() - t
        print(f"[entities_classifier.iter_entities] {len(coref_tokens)} coref tokens [{t} s]")

        t = time.clock()
        #exceptions_i = {-i, k for i, k in enumerate(exceptions.keys())}
        matches = exceptions_matcher(doc)
        t = time.clock() - t

        print(f"[entities_classifier.iter_entities] {len(matches)} matches (pronouns) [{t} s]")
        for _, start, end in exceptions_matcher(doc):
            mention = doc[start:end]
            m_root = mention.root
            if m_root.i in coref_tokens:
                continue
            # assert not m_root._.in_coref, f"{m_root.text!r} in {m_root.coref_clusters[0].main.text!r}"
            m_root_text = m_root.text.lower()
            ent_class = exceptions.get(m_root_text, UNK)

            sent = m_root.sent
            yield {
                'i': mention.start,
                't0': sent.start,
                't1': sent.end,
                #'entity_i': None,
                'entity_root': ent_class.upper(),
                #'entity_pos': None,
                #'entity_tag': None,
                #'entity_text': None,
                'mention_text': mention.text,
                'mention_root': m_root_text,
                'mention_pos': m_root.pos_,
                'mention_tag': m_root.tag_,
                'categ': ent_class,
            }

        #n_mentions = sum(len(c.mentions) for c in doc._.coref_clusters)
        #n_clusters = len(doc._.coref_clusters)
        #print(f"[entities_classifier.iter_entities] {n_mentions} mentions, {n_clusters} clusters")
        for cluster in doc._.coref_clusters:
            e = cluster.main
            e_i = cluster.i
            e_root = e.root
            e_root_text = e_root.text.strip().lower()
            e_pos = e.root.pos_
            e_tag = e.root.tag_
            e_txt = e.text
            for mention in cluster.mentions:
                c_root = e_root or mention.root
                m_root = mention.root
                m_root_text = m_root.text.lower()
                sent = m_root.sent
                #c_root = cluster.main.root if cluster else mention.root
                if c_root.pos_ == 'PROPN':  # resolved to an entity
                    if c_root.ent_type_ == 'PERSON':
                        ent_class = PERSON
                    elif c_root.ent_type_ in {'NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT'}:
                        ent_class = ENV
                    else: # dates, quantities etc.: ignore those
                        ent_class = exceptions.get(m_root_text, NAN)
                elif c_root.pos_ == 'NOUN':  # resolved to a noun
                    hypernyms = {hypernym_map[m.name()] for m in hmatcher(c_root)}
                    if 'person' in hypernyms:
                        ent_class = PERSON
                    else:
                        ent_class = ENV
                elif c_root.pos_ in {'DET', 'PRON'}: # unresolved
                    c_root_text = c_root.text.lower()
                    
                    ret = exceptions.get(c_root_text, None)
                    ret = ret or exceptions.get(m_root_text, None)
                    
                    # unlikely a det/pron doesn't refer to a person (except exceptions)
                    ent_class = ret or PERSON

                else: # verbs, adj etc... most likely irrelevant (or unknown?)
                    ent_class = UNK
                
                yield {
                    'i': mention.start,
                    't0': sent.start,
                    't1': sent.end,
                    'entity_i': e_i,
                    #'entity': e_selected,
                    'entity_root': e_root_text,
                    'entity_pos': e_pos,
                    'entity_tag': e_tag,
                    'entity_text': e_txt,
                    'mention_text': mention.text,
                    'mention_root': m_root_text,
                    'mention_pos': m_root.pos_,
                    'mention_tag': m_root.tag_,
                    'categ': ent_class,
                }
    return iter_entities
    #Doc.set_extension(Doc, doc_attr, method=iter_entities)
