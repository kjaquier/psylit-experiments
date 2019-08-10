import os
import logging
from collections import Counter
from functools import partial

import pandas as pd
import spacy
#from spacy.util import minibatch
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import neuralcoref
#from joblib import Parallel, delayed

from utils import spacy as myspacy
from utils.misc import benchmark, batch

from features import entities as proc_ent
from features import tagging
from features import semantic_parsing as sem

from data import lexicons


logger = logging.getLogger(__name__)


def make_nlp(model='en_core_web_sm'):
    nlp = spacy.load(model)

    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(benchmark(merge_ents), after="ner")

    nlp.add_pipe(benchmark(myspacy.fix_names), after='merge_entities')

    nlp.add_pipe(benchmark(WordnetAnnotator(nlp.lang)), after='tagger')

    coref = neuralcoref.NeuralCoref(nlp.vocab, blacklist=False, store_scores=False, max_dist=20)
    nlp.add_pipe(benchmark(coref), name='neuralcoref')

    nrc_lex = lexicons.load_nrc_wordlevel()
    lextag = tagging.LexiconTagger(nlp, nrc_lex)
    nlp.add_pipe(benchmark(lextag))

    negtag = tagging.NegTagger(nlp.vocab)
    nlp.add_pipe(benchmark(negtag))

    semdep = sem.SemanticDepParser()
    nlp.add_pipe(benchmark(semdep))

    return nlp


class BookParsePipeline:

    def __init__(self, nlp, output_dir, run_name, 
                 batch_size=30_000, minibatch_size=30_000,
                 save_entities=True,
                 save_data=True, save_doc=False, save_features=False):
        self.nlp = nlp
        self.output_dir = output_dir
        self.run_name = run_name
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.save_entities = save_entities
        self.save_data = save_data
        self.save_doc = save_doc
        self.save_features = save_features

        logger.info(f"Pipeline: %s", ', '.join(pname for pname, _ in nlp.pipeline))

        self.doc = None

    def get_output_prefix(self):
        return os.path.join(self.output_dir, self.run_name)

    def parse_batches(self, text):
        output_prefix = self.get_output_prefix()
        texts = batch(text, self.batch_size)

        def parse_texts(nlp, texts):
            for batch_id, doc in enumerate(nlp.pipe(texts)):
                logger.debug("Batch %d: processing...", batch_id)
                yield BookParser(doc)

        logger.debug("Processing...")
        parsers = list(parse_texts(self.nlp, texts))

        if self.save_data:
            filename = f"{output_prefix}.data.csv"
            logger.info("Writing data to %s", filename)
            data_df = pd.concat([p.get_data_df() for p in parsers], axis='rows', sort=False)
            data_df.to_csv(filename)

        if self.save_entities:
            filename = f"{output_prefix}.ent.csv"
            logger.info("Writing entities to %s", filename)
            ent_df = pd.concat([p.get_entities_df() for p in parsers], axis='rows', sort=False)
            # ent_df['batch_id'] = 0
            ent_df.to_csv(filename)

        if self.save_features:
            filename = f"{output_prefix}.feat.csv"
            logger.info("Writing features to %s", filename)
            feat_df = pd.concat([p.get_features_df() for p in parsers], axis='rows', sort=False)
            # feat_df['batch_id'] = 0
            feat_df.to_csv(filename)


    def parse(self, text):
        output_prefix = self.get_output_prefix()
        logger.debug("Processing...")
        doc = self.nlp(text)

        parser = BookParser(doc)
        if self.save_data:
            filename = f"{output_prefix}.data.csv"
            logger.info("Writing data to %s", filename)
            data_df = parser.get_data_df()
            # data_df['batch_id'] = 0
            data_df.to_csv(filename)

        if self.save_entities:
            filename = f"{output_prefix}.ent.csv"
            logger.info("Writing entities to %s", filename)
            ent_df = parser.get_entities_df()
            # ent_df['batch_id'] = 0
            ent_df.to_csv(filename)

        if self.save_features:
            filename = f"{output_prefix}.feat.csv"
            logger.info("Writing features to %s", filename)
            feat_df = parser.get_features_df()
            # feat_df['batch_id'] = 0
            feat_df.to_csv(filename)


class BookParser:

    def __init__(self, doc):
        self.doc = doc

    def get_features_df(self):
        doc = self.doc

        data = [
            {'i': tok.i,
             'sent_i': tok.sent.start,
             't': getattr(tok._.subsent_root, 'i', None),
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

    def get_data_df(self):
        doc = self.doc

        predicates = doc._.lex_matches
        logger.debug(f"{len(predicates)} predicates")
        logger.debug(f"# of duplicates (i): {Counter(Counter(tok.i for tok in predicates).values())}")
        logger.debug(f"# of agents per predicates (incl. None): {Counter(len(tok._.agents or [None]) for tok in predicates)}")
        logger.debug(f"# of patients per predicates (incl. None): {Counter(len(tok._.patients or [None]) for tok in predicates)}")

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
            for agent in (tok._.agents or [None])
            for patient in (tok._.patients or [None])
        ]

        table = pd.DataFrame(data)

        predicate_cols = [c for c in list(table.columns) if c.startswith('L_')]
        table[predicate_cols] = table[predicate_cols].fillna(0)
        return table

    def get_entities_df(self):
        ent_cls = proc_ent.entity_classifier(self.doc.vocab)
        df = pd.DataFrame(ent_cls(self.doc))
        return df
