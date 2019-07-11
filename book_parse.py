import pandas as pd
import numpy as np

import tools
import tools.spacy as myspacy
import tools.pandas as mypd

import processing
import processing.entities as proc_ent
import processing.lexicons as lexicons
import processing.preprocess as preprocess

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 
import neuralcoref

def build_pipe(model='en_core_web_sm'):

    nlp = spacy.load(model)

    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents, after="ner")

    nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')

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

    #ent_cls = proc_ent.entity_classifier(nlp.vocab)
    #nlp.add_pipe(ent_cls)

    return nlp


def get_dataframe(doc):
    fmt = lambda tok: tok.text.strip().lower() if tok else None
    data = [
        {'i': tok.i,
        't': tok.sent.start,
        'neg': tok._.negated,
        'lemma': tok.lemma_[:50],
        'text': fmt(tok)[:50],
        **{('R_'+r): fmt(clust.main.root) for r, clust in tok._.sem_deps}, # FIXME deal with pronouns: join is made on entity_root
                                                                           # which is either clust.main.root or ent_class (Categorical)
        **{('L_'+doc.vocab[cat].text): 1.0 for cat in tok._.lex}, 
        }
        for tok in doc if tok._.has_lex
    ]

    table = pd.DataFrame(data)
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

    ent_rows = []
    for clust in doc._.coref_clusters:
        e = clust.main
        e_i = clust.i
        e_root = e.root.text.strip().lower()
        #e_selected = e.root._.selected_coref_text
        e_pos = e.root.pos_
        e_tag = e.root.tag_
        e_txt = e.text
        for mention in clust.mentions:
            #mlen = len(mention)
            #ner_tags = Counter(f'NER_{t.ent_type_}' for t in mention if t.ent_type_)
            #wn_tags = Counter(f'WN_{ent_type_hypernym_map[m.name()]}' for m in ent_type_matcher(mention.root))
            m_root = mention.root
            sent = mention.root.sent
            ent_rows.append({
                'i': mention.start,
                't0': sent.start,
                't1': sent.end,
                'entity_i': e_i,
                #'entity': e_selected,
                'entity_root': e_root,
                'entity_pos': e_pos,
                'entity_tag': e_tag,
                'entity_text': e_txt,
                'mention_text': mention.text,
                'mention_root': m_root.text.lower(),
                'mention_pos': m_root.pos_,
                'mention_tag': m_root.tag_,
                'categ': mention._.ent_class,
            })
    df = pd.DataFrame(ent_rows)

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
         output_filename:"CSV file to write",
         save_doc:("File where spacy Doc object is saved",'option','d')=None,
         save_entities:("File where entities are saved", 'option','e')=None,
         start:("Position to read from",'option','t0')=None,
         end:("Position to stop reading after",'option','t1')=None):

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

    entities_df = get_entities_df(doc)

    if save_entities:
        entities_df.to_csv(save_entities)

    if save_doc:
        doc.to_disk(save_doc)

    df = get_dataframe(doc)
    print("Data frame size: ", df.size)
    df.to_csv(output_filename)


if __name__ == '__main__':
    import plac
    plac.call(main)