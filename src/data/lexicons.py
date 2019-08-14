import pathlib
from collections import namedtuple

import pandas as pd

LexiconRow = namedtuple('LexiconRow', ['term', 'category', 'weight'])
LEXICONS_ROOT = pathlib.Path('data', 'raw', 'lexicons')


def load_nrc_emotions(prefix=''):
    # with open(LEXICONS_ROOT / 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt') as f:
    #     lines = (x.strip() for x in f.readlines())
    #     lex_records = (x.split('\t') for x in lines if x)
    #     lex_records = [LexiconRow(term, categ, float(flag))
    #                    for term, categ, flag in lex_records]
    
    emlex = pd.read_csv(LEXICONS_ROOT / 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                        sep='\t',
                        header=None,
                        names=['term', 'category', 'weight'])

    emlex = pd.get_dummies(emlex, columns=['category'], prefix=prefix, prefix_sep='')
    weights = emlex.pop('weight')
    emlex_dummies = emlex.multiply(weights, axis='index')
    emlex_dummies.groupby('term').sum()
    return emlex

    
def load_nrc_vad(prefix=''):
    lex = pd.read_csv(LEXICONS_ROOT / 'NRC-VAD-Lexicon.txt', sep='\t')
    column_renaming_map = {'Word': 'term', **{c: prefix + c.lower() for c in lex.columns}}
    lex.rename(index=str, columns=column_renaming_map, inplace=True)
    lex = lex.groupby('term').mean()
    return lex
