from .entities import generate_ent_type

import pandas as pd

def test_env_type():
    
    ents = pd.read_csv(r'data\entities_sample.csv')

    person_toks = ['them', 'her', 'he', 'captain', 'myself', 'ourselves', 'I',
                   'woman', 'man', 'John', 'John Doe', 'Mr. Doe']

    env_toks = ['winter', 'voyage', 'vessel', 'birth', 'abode']

    ents['categ'] = generate_ent_type(ents, level='entity')

    debug_cols = ['mention_root', 'mention_text', 'mention_pos', 'entity_root', 
                  'entity_text', 'entity_pos']

    for _, row in ents[ents.entity_root.isin(person_toks)].iterrows():
        assert row.categ == 'person', row[debug_cols]

    for _, row in ents[ents.entity_root.isin(env_toks)].iterrows():
        assert row.categ == 'environment', row[debug_cols]