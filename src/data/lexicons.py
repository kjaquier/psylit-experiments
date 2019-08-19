import pathlib
from collections import namedtuple

import pandas as pd

from globals import MEMORY


LexiconRow = namedtuple('LexiconRow', ['term', 'category', 'weight'])
LEXICONS_ROOT = pathlib.Path('data', 'raw', 'lexicons')


def load_nrc_emotions(prefix=''):
    emlex = pd.read_csv(LEXICONS_ROOT / 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                        sep='\t',
                        header=None,
                        names=['term', 'category', 'weight'])

    emlex = pd.get_dummies(emlex, columns=['category'], prefix=prefix, prefix_sep='')
    weights = emlex.pop('weight')
    emlex = emlex.multiply(weights, axis='index')
    emlex = emlex.groupby('term').sum()
    emlex.describe()
    return emlex


def load_nrc_vad(prefix=''):
    lex = pd.read_csv(LEXICONS_ROOT / 'NRC-VAD-Lexicon.txt', sep='\t')
    lex.rename(index=str, columns={c: prefix + c.lower() for c in lex.columns}, inplace=True)
    lex.rename(index=str, columns={'word': 'term'}, inplace=True)
    lex.describe()
    lex = lex.groupby('term').mean()
    return lex
