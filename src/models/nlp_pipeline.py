import logging
import json

import pandas as pd
import spacy
import neuralcoref

from utils import spacy as spacy_utils
from utils.misc import benchmark, Timer, path_remove_if_exists

from features import entities as proc_ent
from features import tagging
from features import semantic_parsing as sem

from data import lexicons


logger = logging.getLogger(__name__)


def make_nlp(model='en_core_web_sm', coref_kwargs={}, lexicon_kwargs={}):  # pylint: disable=dangerous-default-value
    nlp = spacy.load(model)

    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents, after="ner")

    nlp.add_pipe(spacy_utils.fix_names, after='merge_entities')

    nlp.add_pipe(spacy_utils.LazyWordnetAnnotator(nlp.lang))
    nlp.add_pipe(proc_ent.EntityTypeHypernymMatcher())

    coref = neuralcoref.NeuralCoref(nlp.vocab, blacklist=False, store_scores=False, **coref_kwargs)
    nlp.add_pipe(benchmark(coref), name='neuralcoref')

    em_lex = lexicons.load_nrc_emotions()
    lextag = tagging.LexiconTagger(nlp, em_lex, **lexicon_kwargs)
    nlp.add_pipe(lextag, name='tag_emotions')

    vad_lex = lexicons.load_nrc_vad()
    lextag = tagging.LexiconTagger(nlp, vad_lex, **lexicon_kwargs)
    nlp.add_pipe(lextag, name='tag_vad')

    negtag = tagging.NegTagger(nlp.vocab)
    nlp.add_pipe(negtag)

    semdep = sem.SemanticDepParser()
    nlp.add_pipe(semdep)

    return nlp


class BookParsePipeline:

    def __init__(self, nlp, 
                 batch_size=30_000,
                 save_entities=True,
                 save_data=True, 
                 save_meta=True, 
                 save_doc=False, 
                 save_features=False):
        self.nlp = nlp
        self.batch_size = batch_size
        self.save_entities = save_entities
        self.save_data = save_data
        self.save_meta = save_meta
        self.save_doc = save_doc
        self.save_features = save_features

        logger.info(f"Pipeline: %s", ', '.join(pname for pname, _ in nlp.pipeline))

        self.data = {}

    def parse_batches(self, texts):

        n_batches = len(texts)

        def parse_texts(nlp, texts):
            last_i = 0
            for batch_id, doc in enumerate(nlp.pipe(texts)):
                logger.info("Processing batch %d / %d", batch_id+1, n_batches)
                yield BookParser(doc, batch_id=batch_id, first_i=last_i)
                last_i += doc[-1].i

        parsers = list(parse_texts(self.nlp, texts))

        if self.save_data:
            data_df = pd.concat([p.get_data_df() for p in parsers], axis='rows', sort=False)
            self.data['data_df'] = data_df

        if self.save_entities:
            ent_df = pd.concat([p.get_entities_df() for p in parsers], axis='rows', sort=False)
            self.data['ent_df'] = ent_df

        if self.save_features:
            feat_df = pd.concat([p.get_features_df() for p in parsers], axis='rows', sort=False)
            self.data['feat_df'] = feat_df

    def save(self, output_dir, run_name, metadata=None):
        assert self.data, "Nothing to save!"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_data:
            out_path = output_dir / f"{run_name}.data.csv"
            logger.info("Writing data to %s", out_path)
            path_remove_if_exists(out_path)
            self.data['data_df'].to_csv(out_path, index=False)
        if self.save_entities:
            out_path = output_dir / f"{run_name}.ent.csv"
            logger.info("Writing entities to %s", out_path)
            path_remove_if_exists(out_path)
            self.data['ent_df'].to_csv(out_path)
        if self.save_features:
            out_path = output_dir / f"{run_name}.feat.csv"
            logger.info("Writing features to %s", out_path)
            path_remove_if_exists(out_path)
            self.data['feat_df'].to_csv(out_path)
        if self.save_meta:
            out_path = output_dir / f"{run_name}.meta.json"
            logger.info("Writing metadata to %s", out_path)
            path_remove_if_exists(out_path)
            with open(out_path, 'w') as f:
                json.dump(metadata, f)
        
    def parse(self, text):
        logger.debug("Processing...")
        doc = self.nlp(text)

        parser = BookParser(doc)
        if self.save_data:
            self.data['data_df'] = parser.get_data_df()

        if self.save_entities:
            self.data['ent_df'] = parser.get_entities_df()

        if self.save_features:
            self.data['feat_df'] = parser.get_features_df()


class BookParser:

    def __init__(self, doc, first_i=0, batch_id=None):
        self.doc = doc
        self.first_i = first_i
        self.batch_id = batch_id

    def get_features_df(self):
        doc = self.doc
        i_0 = self.first_i
        batch_id = self.batch_id
        safe_add_i_0 = lambda i: i_0 + i if i is not None else None

        data = [
            {'i': i_0 + tok.i,
             'batch_id': batch_id,
             'sent_i': i_0 + tok.sent.start,
             't': safe_add_i_0(getattr(tok._.subsent_root, 'i', None)),
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
        i_0 = self.first_i

        predicates = doc._.lex_matches
        n = len(predicates)

        t = Timer()
        t.start()

        data = [
            {'i': i_0 + tok.i,
             'sent_i': i_0 + tok.sent.start,
             't': i_0 + tok._.subsent_root.i,
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

        table = pd.DataFrame(data).sort_values('i')

        predicate_cols = [c for c in list(table.columns) if c.startswith('L_')]
        table[predicate_cols] = table[predicate_cols].fillna(0)

        t.stop()
        logger.debug('%d predicates (%d distinct) [%s]', len(table.index), n, t)

        return table

    def get_entities_df(self):
        t = Timer()
        t.start()
        ent_cls = proc_ent.entity_classifier(self.doc.vocab)
        df = pd.DataFrame(ent_cls(self.doc))
        t.stop()
        logger.debug('%d entities [%s]', len(df.index), t)
        return df
