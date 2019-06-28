from collections import namedtuple
import pandas as pd

NRCWordRow = namedtuple('NRCWordRow', ['term', 'emotion', 'weight'])

def load_nrc_wordlevel():
    with open(r'..\datasets\NRC-Sentiment-Emotion-Lexicons\NRC-Sentiment-Emotion-Lexicons\NRC-Emotion-Lexicon-v0.92\NRC-Emotion-Lexicon-Wordlevel-v0.92.txt') as f:
        lines = (x.strip() for x in f.readlines())
        lex_records = (x.split('\t') for x in lines if x)
        lex_records = [NRCWordRow(term, categ, float(flag)) for term, categ, flag in lex_records ]
    
    

    emlex = pd.DataFrame(lex_records)
    emlex_dummies = pd.get_dummies(pd.Categorical(emlex.emotion), prefix='NRCw_Em')
    emlex_dummies = emlex_dummies.multiply(emlex.weight, axis='index')
    emlex.pop('emotion')
    emlex.pop('weight')
    emlex = pd.concat([emlex, emlex_dummies], axis=1)

    emlex = emlex.groupby('term').sum()
    return emlex
    
    
def load_nrc_senselevel():
    with open(r'..\datasets\NRC-Sentiment-Emotion-Lexicons\NRC-Sentiment-Emotion-Lexicons\NRC-Emotion-Lexicon-v0.92\NRC-Emotion-Lexicon-Senselevel-v0.92.txt') as f:
        lex_records = (x.strip().split('\t') for x in f.readlines())
        lex_records = ((x[0].split("--"), x[1], x[2] == '1') for x in lex_records)
        lex_records = [(x[0][0], [s.strip() for s in x[0][1].split(",")], x[1], x[2]) for x in lex_records]
        return lex_records
    