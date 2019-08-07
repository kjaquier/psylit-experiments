import logging
import sys

import pandas as pd
import spacy

from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import neuralcoref

from utils import spacy as myspacy
from utils.misc import benchmark

from features import entities as proc_ent
from features import tagging
from features import semantic_parsing as sem

from data import lexicons


logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class BookParsePipeline:

    def __init__(self, model='en_core_web_sm'):

        nlp = spacy.load(model)

        merge_ents = nlp.create_pipe("merge_entities")
        nlp.add_pipe(benchmark(merge_ents), after="ner")

        nlp.add_pipe(benchmark(myspacy.fix_names), after='merge_entities')

        nlp.add_pipe(benchmark(WordnetAnnotator(nlp.lang)), after='tagger')

        coref = neuralcoref.NeuralCoref(nlp.vocab, blacklist=False, max_dist=20, store_scores=False)
        nlp.add_pipe(benchmark(coref), name='neuralcoref')

        nrc_lex = lexicons.load_nrc_wordlevel()
        lextag = tagging.LexiconTagger(nlp, nrc_lex)
        nlp.add_pipe(benchmark(lextag))

        negtag = tagging.NegTagger(nlp.vocab)
        nlp.add_pipe(benchmark(negtag))

        semdep = sem.SemanticDepParser()
        nlp.add_pipe(benchmark(semdep))

        self.nlp = nlp
        self.doc = None

    def parse(self, text):
        self.doc = self.nlp(text)

    def get_features_df(self):
        doc = self.doc
        if not doc:
            raise Exception('Need to run parse() first!')
        data = [
            {'i': tok.i,
             'sent_i': tok.sent.start,
             't': getattr(tok._.subsent_root, 'i', tok.sent.start),
             'neg': tok._.negated,
             'lemma': tok.lemma_,
             'text': tok.text,
             'dep': tok.dep_,
             'pos': tok.pos_,
             'agents': ','.join(t.root.text for t in tok._.agents) or None,
             'patients': ','.join(t.root.text for t in tok._.patients) or None,
             'lex': ','.join(doc.vocab[cat].text for cat in tok._.lex) or None}
            for tok in doc
        ]

        table = pd.DataFrame(data)
        return table

    def get_df(self):
        doc = self.doc
        if not doc:
            raise Exception('Need to run parse() first!')

        predicates = doc._.lex_matches
        data = [
            {'i': tok.i,
             'sent_i': tok.sent.start,
             't': tok._.subsent_root.i,
             'neg': tok._.negated,
             'lemma': tok.lemma_,
             'text': tok.text,
             'R_agent': agent.root.text if agent else None,
             'R_patient': patient.root.text if patient else None,
             **{('L_'+doc.vocab[cat].text): 1.0 for cat in tok._.lex},
             }
            for tok in predicates
            for agent in tok._.agents
            for patient in (tok._.patients or [None])
        ]

        table = pd.DataFrame(data)

        predicate_cols = [c for c in list(table.columns) if c.startswith('L_')]
        table[predicate_cols] = table[predicate_cols].fillna(0)
        return table

    def get_entities_df(self):
        doc = self.doc
        if not doc:
            raise Exception('Need to run parse() first!')

        ent_cls = proc_ent.entity_classifier(doc.vocab)
        df = pd.DataFrame(ent_cls(doc))
        return df
