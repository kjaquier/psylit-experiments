import logging
import time
import sys
import json
import os

import tools
import tools.spacy as myspacy
import tools.pandas as mypd

import processing
import processing.entities as proc_ent
import processing.lexicons as lexicons
import processing.preprocess as preprocess
import processing.doc2graph as d2g

import pandas as pd
import numpy as np

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 
import neuralcoref

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def benchmark(f):
    rename = True
    try:
        name = f.__name__
    except AttributeError:
        try:
            name = f.__class__.__name__
        except AttributeError:
            name = repr(f)
            rename = False

    def bench_f(*args, **kwargs):
        t = time.clock()
        res = f(*args, **kwargs)
        t = time.clock() - t
        print(f"[Time] {name} {t*1000:.3n}ms")
        return res

    if rename:
        bench_f.__name__ = name

    return bench_f

def build_pipe(model='en_core_web_sm'):

    nlp = spacy.load(model)

    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(benchmark(merge_ents), after="ner")

    nlp.add_pipe(benchmark(myspacy.fix_names), after='merge_entities')

    nlp.add_pipe(benchmark(WordnetAnnotator(nlp.lang)), after='tagger')

    coref = neuralcoref.NeuralCoref(nlp.vocab, blacklist=False, max_dist=20, store_scores=False)
    nlp.add_pipe(benchmark(coref), name='neuralcoref')

    #coref_lu = myspacy.CorefLookup()
    #nlp.add_pipe(benchmark(coref_lu))

    nrc_lex = lexicons.load_nrc_wordlevel()
    lextag = myspacy.LexiconTagger(nlp, nrc_lex)
    nlp.add_pipe(benchmark(lextag))

    negtag = myspacy.NegTagger(nlp.vocab)
    nlp.add_pipe(benchmark(negtag))

    semdep = myspacy.SemanticDepParser()
    nlp.add_pipe(benchmark(semdep))

    #predparser = myspacy.PredicateParser(nlp.vocab)
    #nlp.add_pipe(benchmark(predparser))

    #ent_cls = proc_ent.entity_classifier(nlp.vocab)
    #nlp.add_pipe(benchmark(ent_cls))

    return nlp


def get_dataframe(doc):
    #g = d2g.DocGraph(doc)
    # fmt = lambda txt: txt.strip().lower()[:50]
    # data = [
    #     {
    #         't': t,
    #         'i': predicate.i,
    #         'neg': predicate._.negated,
    #         'lemma': fmt(predicate.lemma_),
    #         'text': fmt(predicate.text),
    #         'R_agent': fmt(agent.root.text) if agent else None,
    #         'R_patient': fmt(patient.root.text) if patient else None,
    #         **{('L_'+doc.vocab[cat].text): 1.0 for cat in predicate._.lex}, 
    #     }
    #     for t, predicate, agent, patient in g.iter_frames()
    # ]

    #fmt = lambda tok: tok.text.strip().lower() if tok else None
    data = [
        {'i': tok.i,
        't': tok.sent.start,
        'neg': tok._.negated,
        'lemma': tok.lemma_,
        'text': tok.text,
        'R_agent': agent.root.text if agent else None,
        'R_patient': patient.root.text if patient else None,
        #**{('R_'+r): fmt(clust.main.root) for r, clust in tok._.sem_deps}, # FIXME deal with pronouns: join is made on entity_root
        #                                                                   # which is either clust.main.root or ent_class (Categorical)
        **{('L_'+doc.vocab[cat].text): 1.0 for cat in tok._.lex}, 
        }
        for tok in doc._.lex_matches #doc if tok._.has_lex
        for agent in tok._.agents
        for patient in (tok._.patients or [None])
    ]

    table = pd.DataFrame(data)

    #for col in ('lemma', 'text', 'R_agent', 'R_patient'):
    #    print('Formatting', col, table[col].dtype)
    #    table[col] = table[col].fillna('').str.strip().str.lower().str.slice(stop=50)

    
    rel_cols = table.columns[list(table.columns.str.startswith('R_'))]
    lex_cols = table.columns[list(table.columns.str.startswith('L_'))]
    table[lex_cols] = table[lex_cols].fillna(0)
    return table

# def generate_ent_type(ents, level='entity'):
    #     n = len(ents.index)
        
    #     pos = ents['entity_pos' if level == 'entity' else 'mention_pos']
        
    #     is_relevant = pos.isin(['PROPN','NOUN','DET','PRON'])
    #     resolved = pos == 'PROPN'
    #     is_noun = pos == 'NOUN'
    #     is_irrelevant = ~is_relevant | (ents.NER_CARDINAL > 0) | (ents.NER_DATE > 0)
        
    #     is_noun_for_person = is_noun & (ents.WN_person > 0) 
    #     is_person = resolved | is_noun_for_person
    #     is_unknown = ents.entity_pos.isin(['DET','PRON'])

    #     ent_type = pd.Series(np.array(['person'] * n))
    #     ent_type[~is_person] = 'environment'
    #     ent_type[is_unknown] = 'unknown'
    #     ent_type[is_irrelevant] = None
    #     ent_type = pd.Categorical(ent_type)
    #     return ent_type


def get_entities_df(doc):

    ent_cls = proc_ent.entity_classifier(doc.vocab)

    df = pd.DataFrame(ent_cls(doc))
    return df 

# def get_entities_df(doc):
    # ent_rows = []
    # for clust in doc._.coref_clusters:
    #     e = clust.main
    #     e_i = clust.i
    #     e_root = e.root.text.strip().lower()
    #     #e_selected = e.root._.selected_coref_text
    #     e_pos = e.root.pos_
    #     e_tag = e.root.tag_
    #     e_txt = e.text
    #     for mention in clust.mentions:
    #         #mlen = len(mention)
    #         #ner_tags = Counter(f'NER_{t.ent_type_}' for t in mention if t.ent_type_)
    #         #wn_tags = Counter(f'WN_{ent_type_hypernym_map[m.name()]}' for m in ent_type_matcher(mention.root))
    #         m_root = mention.root
    #         sent = mention.root.sent
    #         ent_rows.append({
    #             'i': mention.start,
    #             't0': sent.start,
    #             't1': sent.end,
    #             'entity_i': e_i,
    #             #'entity': e_selected,
    #             'entity_root': e_root,
    #             'entity_pos': e_pos,
    #             'entity_tag': e_tag,
    #             'entity_text': e_txt,
    #             'mention_text': mention.text,
    #             'mention_root': m_root.text.lower(),
    #             'mention_pos': m_root.pos_,
    #             'mention_tag': m_root.tag_,
    #             'categ': mention._.ent_class,
    #         })
    # df = pd.DataFrame(ent_rows)

    # ner_cols = df.columns[list(df.columns.str.startswith('NER_'))]
    # df[ner_cols] = df[ner_cols].fillna(0)
    # wn_cols = df.columns[list(df.columns.str.startswith('WN_'))]
    # df[wn_cols] = df[wn_cols].fillna(0)
    
    # narrator = df.categ == 'narrator'
    # reader = df.categ == 'reader'
    
    # df.loc[narrator,'entity_root'] = 'NARRATOR'
    # df.loc[reader,'entity_root'] = 'READER'
    
    # df.loc[narrator,'categ'] = 'person'
    # df.loc[reader,'categ'] = 'person'
    
    return df


def main(input_filename:"Raw text of book to read (UTF-8)",
         run_name:"Name of the run for output files",
         output_dir:"Folder to write output files",
         save_doc:("File where spacy Doc object is saved", 'flag', 'd')=False,
         save_entities:("File where entities are saved", 'flag', 'e')=False,
         save_meta:("Write metadata file about the run", 'flag', 'm')=False,
         start:("Position to read from",'option','t0')=None,
         end:("Position to stop reading after",'option','t1')=None,
         benchmark:("Measure execution time", 'flag', 'b')=False):


    print('start=', repr(start), ' end=', repr(end))
    start = int(start) if start is not None else 0
    end = int(end) if end is not None else None

    if benchmark:
        t0 = time.clock()
    txt = preprocess.read_pg(input_filename)
    n = len(txt)
    txt = txt[start:] if end is None else txt[start:end]
    print('Processing', len(txt), '/', n, 'chars')
    nlp = build_pipe()
    print("Pipeline: ", ', '.join(pname for pname, _ in nlp.pipeline))

    if benchmark:
        t_init = time.clock() - t0
        print(f"Read and pipeline init time: {t_init*1000:.5n}ms")
        t1 = time.clock()

    doc = nlp(txt)
    entities_df = get_entities_df(doc)
    df = get_dataframe(doc)

    if benchmark:
        t_process = time.clock() - t1
        print(f"Process time: {t_process*1000:.5n}ms")
        t2 = time.clock()

    ent_file = os.path.join(output_dir, run_name) + '.ent.csv'
    doc_file = os.path.join(output_dir, run_name) + '.doc.pkl'
    data_file = os.path.join(output_dir, run_name) + '.data.csv'
    meta_file = os.path.join(output_dir, run_name) + '.meta.json'

    if save_entities:
        print(f"Saving entities")
        entities_df.to_csv(ent_file)

    if save_doc:
        print(f"Saving doc object")
        doc.to_disk(save_doc)

    print(f"Saving data")
    df.to_csv(data_file)

    if benchmark:
        t_write = time.clock() - t2
        print(f"Write time: {t_write*1000:.5n}ms")

    if save_meta:
        metadata = {
            'cmd': sys.argv,
            'input_filename': input_filename,
            'time_init': t_init if benchmark else None,
            'time_process': t_process if benchmark else None,
            'time_write': t_write if benchmark else None,
            'n_predicates': len(df.index),
            'n_corefs': len(entities_df.index),
            'ent_file': ent_file,
            'doc_file': doc_file,
            'data_file': data_file,
        }

        for k in ['n_predicates', 'n_corefs']:
            print(k, ':', metadata[k])

        print(f"Saving meta data")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)


if __name__ == '__main__':
    import plac
    plac.call(main)