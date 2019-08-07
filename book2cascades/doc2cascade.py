import os

from statistics import mean

from os import linesep as EOL

import re
import json

import pandas as pd
#import numpy as np

from .multicascade import MultiCascade

# Pandas utils

def df_map_columns(data, cols, lookup, names=None, replace_cols=True):
    merged = data.copy()
    names = names or cols
    for col, col_new_name in zip(cols, names):
        merged_col = merged[[col]].merge(lookup, how='left', left_on=col, right_index=True)#, suffixes=suffixes)

        if replace_cols:
            merged.pop(col)
            
        merged[col_new_name] = merged_col[lookup.name]
        
    return merged

# 

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
        self.data = df.groupby(['t','R_agent','R_patient','lemma']).mean().reset_index()

        self.ent_counts = None

    def _format_rel_cols(self, df):
        dfr = df[self.rel_cols]
        dfr = dfr.fillna('NONE')
        dfr = dfr.replace('NONE','')
        dfr = dfr.apply(lambda c: c.str.lower(), axis='columns')
        dfr = dfr.replace('','NONE')
        
        dfc = df.copy()
        dfc.loc[:, self.rel_cols] = dfr
        return dfc

    def most_common_ents(self):
        if not self.ent_counts:
            self.ent_counts = pd.concat([self.data[r] for r in self.rel_cols], axis='rows').replace('NONE',None).dropna().value_counts()
            #self.most_common = ent_counts.idxmax() # most common
        return self.ent_counts


    def str_match(self, col, pattern):
        return self.data[self.data[col].str.lower().str.match(pattern)]

    def _make_entity_lookup(self, subject=None):
        ents_lookup = self.ents[['t0','entity_root', 'categ']].drop_duplicates()
        #ents_lookup.rename(index=str, columns={'entity_root': 'root', 'entity_name': 'name'}, inplace=True)
        #ents_lookup.set_index('entity_root', inplace=True, verify_integrity=True)
        
        is_narrator = ents_lookup.categ == 'narrator'
        is_reader = ents_lookup.categ == 'reader'

        if subject:
            is_subject = ents_lookup.entity_root == subject
        else:
            is_subject = is_narrator
        
        #assert is_subject.any(), f"No occurence found for entity '{subject}'"

        #ents_lookup.loc[narrator,'entity_root'] = 'NARRATOR'
        #ents_lookup.loc[reader,'entity_root'] = 'READER'

        ents_lookup.loc[is_narrator,'categ'] = 'person'
        ents_lookup.loc[is_reader,'categ'] = 'person'
        
        ents_lookup.loc[is_subject, 'categ'] = 'subject'
        
        ents_lookup.loc[is_subject, 'categ'] = 'subject'
        
        
        most_likely_categ = lambda fr: fr.categ.value_counts(normalize=True, ascending=False, dropna=False).idxmax()
        ents_lookup = ents_lookup.groupby('entity_root').apply(most_likely_categ)
        ents_lookup.name = 'ent_class'
        
        ents_lookup.loc['NONE'] = 'none'
        
        #ents_lookup.at['NARRATOR']
        return ents_lookup

    def _per_entity_data(self, take_top=10):
        ent_counts = pd.concat([self.data[r] for r in self.rel_cols], axis='rows').replace('NONE',None).dropna().value_counts()
        selected_ents = ent_counts.sort_values(ascending=False)[:take_top].index
        mapped = []
        for subject in selected_ents:
            ents_lookup = self._make_entity_lookup(subject=subject)
            subj_data = df_map_columns(self.data, self.rel_cols, ents_lookup)
            subj_data['Subject'] = subject
            mapped.append(subj_data)
            
        return pd.concat(mapped, axis='rows').groupby('Subject')
    
    def _get_cascades_for_entity(self, casc, entity=None):
        ents_lookup = self._make_entity_lookup(entity)
        merged = df_map_columns(casc, self.rel_cols, ents_lookup)

        def cascade_representation(data, symbolic_cols, numeric_cols, symbolic_na=False, casc_index=['t']):
            sym_cascades = pd.get_dummies(data[symbolic_cols], dummy_na=symbolic_na)
            num_cascades = (data[numeric_cols] > data[numeric_cols].mean()) * 1
            casc = pd.concat([data[casc_index], sym_cascades, num_cascades], axis='columns')
            casc = casc.groupby(casc_index).any()*1

            return casc
            
        # Discard NaNs, negations and irrelevant columns
        keep = (~merged[self.rel_cols].isna()).any(axis='columns') & ~merged.neg.fillna(False)
        casc = merged[keep][['t','neg']+self.rel_cols+self.lex_cols]

        # Transform into cascades
        casc = cascade_representation(casc,
                                      symbolic_cols=self.rel_cols,
                                      numeric_cols=self.lex_cols)


        # Drop irrelevant columns
        irrelevant_cols = [col for col in casc.columns if col[1] in ('none',)]
        casc = casc.drop(irrelevant_cols, axis='columns')


        # Put all unknown into a single columns
        unk_cols = [c for c in casc.columns if c.endswith('unknown')]
        casc['R_unknown'] = casc[unk_cols].any(axis='columns').astype(int)
        casc.drop(unk_cols, axis='columns', inplace=True)

        return casc

    def get_all_cascades(self):
        grouped_data = self._per_entity_data()
        cascades = []
        # TODO this can be optimised with e.g Joblib
        for ent, df in grouped_data:
            casc = self._get_cascades_for_entity(df, ent)
            cascades.append((ent, casc))
        return dict(cascades)

