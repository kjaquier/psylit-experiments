import pathlib
import logging
from functools import partial

import json

import pandas as pd
import numpy as np
#from joblib import Parallel, delayed

from utils.pandas import df_map_columns
from features.entities import EXCEPTIONS as ENTITY_EXCEPTIONS

logger = logging.getLogger(__name__)

NONE_PLACEHOLDER = 'NONE'  # necessary because pandas .groupby() doesn't properly handle it


def make_entity_lookup(ents):
    entity_not_found = ents.entity_i.isna()
    ents_fixed = ents.copy()
    entity_not_found_ids = -ents[entity_not_found].mention_root.factorize()[0]

    ents_fixed.loc[entity_not_found, 'entity_i'] = entity_not_found_ids
    ents_fixed.entity_i = ents_fixed.entity_i.astype(np.int32)

    ents_lookup = ents_fixed[['entity_i', 'entity_root', 'categ']].drop_duplicates()
    n = len(ents_lookup.index)

    exceptions_lookup = pd.Series(ENTITY_EXCEPTIONS)
    ents_lookup = ents_lookup.append([exceptions_lookup], sort=True)

    logger.debug("Added %d entities from exception list", len(ents_lookup.index) - n)

    #for ent_txt, ent_class in ENTITY_EXCEPTIONS.items():
        #logger.debug("ents_lookup.at[%s]=[%s]", ent_txt, ent_class)
    #    ents_lookup.at[ent_txt] = ent_class

    #ents_lookup.rename(index=str, columns={'entity_root': 'root', 'entity_name': 'name'}, inplace=True)
    #ents_lookup.set_index('entity_root', inplace=True, verify_integrity=True)
    
    ents_lookup.loc[ents_lookup.categ == 'narrator', 'categ'] = 'person'
    ents_lookup.loc[ents_lookup.categ == 'reader', 'categ'] = 'person'
    
    most_likely_categ = lambda fr: fr.categ.value_counts(normalize=True, ascending=False, dropna=False).idxmax()
    ents_lookup = ents_lookup.groupby('entity_root').apply(most_likely_categ)
    ents_lookup.name = 'ent_class'
    
    ents_lookup.loc[NONE_PLACEHOLDER] = None

    return ents_lookup


def update_lookup_with_subject(ents_lookup, subject):
    ents_lookup = ents_lookup.copy()
    ents_lookup.at[subject] = 'subject'
    return ents_lookup


def cascade_representation(data, symbolic_cols, numeric_cols, casc_index=('t',), **kwargs):
    casc_index = list(casc_index)
    casc = pd.get_dummies(data, columns=symbolic_cols, dummy_na=True, dtype=np.uint8, **kwargs)
    casc.loc[:, numeric_cols] = (data[numeric_cols] > data[numeric_cols].mean())
    casc = casc.groupby(casc_index).any()
    
    return casc


class BookData:

    def __init__(self, data_file, ent_file, meta_file):
        self.ents = pd.read_csv(ent_file, index_col=0)
        df = pd.read_csv(data_file, index_col=0)

        self.rel_cols = list(df.columns[list(df.columns.str.startswith('R_'))])
        self.lex_cols = list(df.columns[list(df.columns.str.startswith('L_'))])
        with open(meta_file) as f:
            self.meta = json.load(f)

        df = self._format_rel_cols(df.drop_duplicates())
        self.data = df.groupby(
            ['t', 'R_agent', 'R_patient', 'lemma']).mean().reset_index()

    def _format_rel_cols(self, df):
        dfr = df[self.rel_cols]
        dfr = dfr.fillna(NONE_PLACEHOLDER)
        dfr = dfr.replace(NONE_PLACEHOLDER, '')
        dfr = dfr.apply(lambda c: c.str.lower(), axis='columns')
        dfr = dfr.replace('', NONE_PLACEHOLDER)

        dfc = df.copy()
        dfc.loc[:, self.rel_cols] = dfr
        return dfc

    def str_match(self, col, pattern):
        return self.data[self.data[col].str.lower().str.match(pattern)]

    def get_all_cascades(self, min_entities_occurrences=50):
        relevant_columns = ['t', 'neg'] + self.rel_cols + self.lex_cols
        data = self.data[relevant_columns]

        ent_counts = pd.concat([data[r] for r in self.rel_cols], axis='rows')
        ent_counts = ent_counts.replace(NONE_PLACEHOLDER, None).dropna()
        ent_counts = ent_counts.value_counts()
        selected_ents_count = ent_counts[ent_counts > min_entities_occurrences]
        logger.debug("%d entities selected (occurring >= %d times)", len(selected_ents_count.index), min_entities_occurrences)
        selected_ents = selected_ents_count.index

        ents_lookup = make_entity_lookup(self.ents)

        def gen_cascades_for_subject(data, ents_lookup, rel_cols, lex_cols, subject):
            ents_lookup = update_lookup_with_subject(ents_lookup, subject)
            
            # Only consider rows between first and last occurrences
            # Edit: as a second thought, it might actually screw up transfer entropy
            # subj_occs = data[rel_cols].apply(lambda c: c == subject).any(axis='columns')
            # subj_occs = subj_occs.where(subj_occs)
            # first_occ = subj_occs.first_valid_index()
            # last_occ = subj_occs.last_valid_index()
            # data = data.loc[first_occ:last_occ, :]

            subj_data = df_map_columns(data, rel_cols, ents_lookup)
            
            subj_casc = cascade_representation(subj_data,
                                               symbolic_cols=rel_cols,
                                               numeric_cols=lex_cols)

            return subject, subj_casc
        
        executor = list #Parallel(n_jobs=-1)
        do = partial(gen_cascades_for_subject, data, ents_lookup, self.rel_cols, self.lex_cols) # delayed()
        tasks = (do(subject) for subject in selected_ents if ents_lookup.get(subject, '') == 'person')
        logger.info("Generating cascades for %d entities (max len = %d)", len(selected_ents), len(self.data.index))
        subj_cascades = executor(tasks)
        
        subj_names = [x for x, _ in subj_cascades]
        subj_cascades = [x for _, x in subj_cascades]
        
        casc = pd.concat(subj_cascades, keys=subj_names, names=['Subject'], sort=False)#, ignore_index=True)

        # Because all relations don't necessarily occur for all subjects, they are set to NaN
        # in concat
        binary_cols = [c for c in casc.columns if c.startswith(('R_', 'L_', 'neg'))]
        casc.loc[:, binary_cols] = casc[binary_cols].fillna(0).astype(np.int8)

        # Drop irrelevant columns
        #irrelevant_cols = [col for col in casc.columns if col[1] in ('none',)]
        #casc = casc.drop(irrelevant_cols, axis='columns')

        # Put all unknowns into a single columns
        unk_cols = [c for c in casc.columns if c.endswith('unknown')]
        casc['R_unknown'] = casc[unk_cols].any(axis='columns').astype(np.uint8)
        casc.drop(unk_cols, axis='columns', inplace=True)
        
        return casc
    
