import logging
from functools import partial

import pandas as pd
import numpy as np

from utils.pandas import df_map_columns
from features.entities import EXCEPTIONS as ENTITY_MANUAL_RESOLUTION

logger = logging.getLogger(__name__)

NONE_PLACEHOLDER = 'NONE'  # necessary because pandas .groupby() doesn't properly handle it

MANUAL_RESOLUTION_LOOKUP_DICT = {**ENTITY_MANUAL_RESOLUTION,
                                 **{v: v for v in ENTITY_MANUAL_RESOLUTION.values()}}
MANUAL_RESOLUTION_LOOKUP = pd.Series(MANUAL_RESOLUTION_LOOKUP_DICT)
MANUAL_RESOLUTION_LOOKUP.name = 'ent_class'


def make_entity_lookup(ents):
    # Create a unique entity_i for those that don't have any
    # Note: not needed since not used to distinguish entities 
    # (could be, but doesn't seem much of a benefit as the ambiguitious
    #  are unlikely to occur sufficiently to be considered in the analysis)
    # entity_not_found = ents.entity_i.isna()
    # ents_fixed = ents.copy()
    # entity_not_found_ids = -ents[entity_not_found].mention_root.factorize()[0]
    # ents_fixed.loc[entity_not_found, 'entity_i'] = entity_not_found_ids
    # ents_fixed.entity_i = ents_fixed.entity_i.astype(np.int32)

    most_likely_categ = lambda fr: fr.categ.value_counts(normalize=True, ascending=False, dropna=False).idxmax()
    ents_lookup = ents.groupby('entity_root').apply(most_likely_categ)

    idx_diff = MANUAL_RESOLUTION_LOOKUP.index.difference(ents_lookup.index)
    ents_lookup = ents_lookup.append(MANUAL_RESOLUTION_LOOKUP[idx_diff].str.lower())
    
    # replaces categories 'narrator' and 'readers' with 'person'
    ents_lookup.replace(['narrator', 'reader'], 'person', inplace=True)
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

    def __init__(self, data_df, ents_df, **metadata):
        self.ents = ents_df
        self.meta = metadata

        self.rel_cols = list(data_df.columns[list(data_df.columns.str.startswith('R_'))])
        self.lex_cols = list(data_df.columns[list(data_df.columns.str.startswith('L_'))])

        df = self._format_rel_cols(data_df.drop_duplicates())
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
        relevant_data_columns = ['t', 'neg'] + self.rel_cols + self.lex_cols

        ents_lookup = make_entity_lookup(self.ents)
        
        # Resolve entities and data with the manual rules before counting them
        ents_resolved = self.ents.copy()
        ents_resolved.entity_root.replace(MANUAL_RESOLUTION_LOOKUP_DICT, inplace=True)
        data_resolved = self.data[relevant_data_columns].copy()
        for col in self.rel_cols:
            data_resolved[col] = data_resolved[col].str.lower().replace(MANUAL_RESOLUTION_LOOKUP_DICT)

        # Count entities and select subjects based on the frequency threshold
        subjects = ents_resolved[~ents_resolved.entity_root.isin({'ENVIRONMENT', 'UNKNOWN', 'PERSON'})]
        logger.debug('%d subjects / %d entities', len(subjects.index), len(ents_resolved.index))
        subjects_freq = (
            subjects
            .groupby('entity_root')
            .size()
        )
        selected_subjects_freq = subjects_freq[subjects_freq > min_entities_occurrences]
        selected_subjects = selected_subjects_freq.index.values
        n = len(selected_subjects)
        logger.debug("selected subjects (occurrences): \n%s", selected_subjects_freq)
        logger.info("%d subjects selected (occurring >= %d times)", n, min_entities_occurrences)

        # Iterate over selected subjects to generate the cascades
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
        do = partial(gen_cascades_for_subject, data_resolved, ents_lookup, self.rel_cols, self.lex_cols) # delayed()
        tasks = (do(subject) for subject in selected_subjects if ents_lookup.get(subject, '') == 'person')
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
    
