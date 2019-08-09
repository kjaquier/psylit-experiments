import os
import logging
from functools import partial

import json

import pandas as pd
import numpy as np
#from joblib import Parallel, delayed

from utils.pandas import df_map_columns


logger = logging.getLogger(__name__)


def make_entity_lookup(ents):
    entity_not_found = ents.entity_i.isna()
    ents_fixed = ents.copy()
    entity_not_found_ids = -ents[entity_not_found].mention_root.factorize()[0]

    ents_fixed.loc[entity_not_found, 'entity_i'] = entity_not_found_ids
    ents_fixed.entity_i = ents_fixed.entity_i.astype(np.int32)

    ents_lookup = ents_fixed[['entity_i', 'entity_root', 'categ']].drop_duplicates()
    #ents_lookup.rename(index=str, columns={'entity_root': 'root', 'entity_name': 'name'}, inplace=True)
    #ents_lookup.set_index('entity_root', inplace=True, verify_integrity=True)
    
    ents_lookup.loc[ents_lookup.categ == 'narrator', 'categ'] = 'person'
    ents_lookup.loc[ents_lookup.categ == 'reader', 'categ'] = 'person'
    
    most_likely_categ = lambda fr: fr.categ.value_counts(normalize=True, ascending=False, dropna=False).idxmax()
    ents_lookup = ents_lookup.groupby('entity_root').apply(most_likely_categ)
    ents_lookup.name = 'ent_class'
    
    ents_lookup.loc['NONE'] = 'none'

    return ents_lookup


def update_lookup_with_subject(ents_lookup, subject):
    ents_lookup = ents_lookup.copy()
    ents_lookup.at[subject] = 'subject'
    return ents_lookup


def cascade_representation(data, symbolic_cols, numeric_cols, symbolic_na=False, casc_index=('t',)):
    casc_index = list(casc_index)
    sym_cascades = pd.get_dummies(data[symbolic_cols], dummy_na=symbolic_na)
    num_cascades = (data[numeric_cols] > data[numeric_cols].mean()) * 1  # NaN -> 0
    casc = pd.concat([data[casc_index], sym_cascades, num_cascades], axis='columns')
    casc = casc.groupby(casc_index).any()*1

    return casc


class BookData:

    def __init__(self, book_name, data_folder):
        filename = lambda folder, name, ext: os.path.join(folder, f"{name}.{ext}")

        self.ents = pd.read_csv(filename(data_folder, book_name, "ent.csv"), index_col=0)
        df = pd.read_csv(filename(data_folder, book_name, "data.csv"), index_col=0)
        meta_file = filename(data_folder, book_name, "meta.json")

        self.rel_cols = list(df.columns[list(df.columns.str.startswith('R_'))])
        self.lex_cols = list(df.columns[list(df.columns.str.startswith('L_'))])
        with open(meta_file) as f:
            self.meta = json.load(f)

        df = self._format_rel_cols(df.drop_duplicates())
        self.data = df.groupby(
            ['t', 'R_agent', 'R_patient', 'lemma']).mean().reset_index()

        self.ent_counts = None

    def _format_rel_cols(self, df):
        dfr = df[self.rel_cols]
        dfr = dfr.fillna('NONE')
        dfr = dfr.replace('NONE', '')
        dfr = dfr.apply(lambda c: c.str.lower(), axis='columns')
        dfr = dfr.replace('', 'NONE')

        dfc = df.copy()
        dfc.loc[:, self.rel_cols] = dfr
        return dfc

    def most_common_ents(self):
        if not self.ent_counts:
            self.ent_counts = pd.concat([self.data[r] for r in self.rel_cols], axis='rows')
            self.ent_counts = self.ent_counts.replace('NONE', None).dropna().value_counts()
            
        return self.ent_counts

    def str_match(self, col, pattern):
        return self.data[self.data[col].str.lower().str.match(pattern)]

    def get_all_cascades(self, n_entities=10, group=True):
        relevant_columns = ['t', 'neg'] + self.rel_cols + self.lex_cols
        data = self.data[relevant_columns]

        ent_counts = pd.concat([data[r] for r in self.rel_cols], axis='rows')
        ent_counts = ent_counts.replace('NONE', None).dropna().value_counts()
        selected_ents_count = ent_counts.sort_values(ascending=False)[:n_entities]
        logger.debug("Selected entities: %s", selected_ents_count)
        selected_ents = selected_ents_count.index

        ents_lookup = make_entity_lookup(self.ents)

        def gen_cascades_for_subject(data, ents_lookup, rel_cols, lex_cols, subject):
            ents_lookup = update_lookup_with_subject(ents_lookup, subject)

            subj_occs = data[rel_cols].apply(lambda c: c == subject).any(axis='columns')
            subj_occs = subj_occs.where(subj_occs)
            first_occ = subj_occs.first_valid_index()
            last_occ = subj_occs.last_valid_index()
            data = data.loc[first_occ:last_occ, :]

            subj_data = df_map_columns(data, rel_cols, ents_lookup)
            subj_casc = cascade_representation(subj_data,
                                               symbolic_cols=rel_cols,
                                               numeric_cols=lex_cols)

            subj_casc['Subject'] = subject
            return subj_casc
        
        executor = list #Parallel(n_jobs=-1)
        do = partial(gen_cascades_for_subject, self.data, ents_lookup, self.rel_cols, self.lex_cols) # delayed()
        tasks = (do(subject) for subject in selected_ents)
        logger.info("Generating cascades for %d entities (max len = %d)", len(selected_ents), len(self.data.index))
        subj_cascades = executor(tasks)
        
        casc = pd.concat(subj_cascades, axis='rows', sort=False)

        # Drop irrelevant columns
        irrelevant_cols = [col for col in casc.columns if col[1] in ('none',)]
        casc = casc.drop(irrelevant_cols, axis='columns')

        # Put all unknowns into a single columns
        unk_cols = [c for c in casc.columns if c.endswith('unknown')]
        casc['R_unknown'] = casc[unk_cols].any(axis='columns').astype(int)
        casc.drop(unk_cols, axis='columns', inplace=True)

        if group:
            casc = casc.groupby('Subject')

        return casc
    
