from collections import Counter, defaultdict
from os import linesep as EOL
from itertools import islice

from . import cleanup

def read_pg(filename):
    with open(filename, encoding='utf8') as f:
        lines = cleanup.filter_headers(f.readlines())
        lines = (l.lstrip() for l in lines)
        whole_txt = '\n'.join(l for l in lines if l)
        whole_txt = whole_txt.replace('—','')
    return whole_txt


POS_SPACY_TO_WN = {
    'ADJ':   'a', # adjective (e.g: big, old, green, incomprehensible, first)
    'ADV':   'r', # adverb (e.g: very, tomorrow, down, where, there)
    'VERB':  'v', # verb (e.g: run, runs, running, eat, ate, eating)
}

class PsychFeatsExtractor:
    """Extracts knowledge-based psychological features"""

    def __init__(self):
        self.counters = defaultdict(Counter)

    def __call__(self, tokens):
        for tok in tokens:
            psy_synsets = frozenset([
                s.name() 
                for s in tok._.wordnet.wordnet_synsets_for_domain(['psychological_features'])
            ])
            if psy_synsets:
                self.counters['synsets'][psy_synsets] += 1
                yield psy_synsets
            elif tok.ent_iob_ in "BI":
                self.counters['entities'][tok.text] += 1
                yield frozenset([tok.text])
            else:
                txt = tok.text.lower()
                self.counters['others'][txt] += 1
                yield frozenset([txt])

    def stats(self):
        return {
            'n_synsets': sum(self.counters['synsets'].values()),
            'n_entities': sum(self.counters['entities'].values()),
            'n_others': sum(self.counters['others'].values()),
            'entities': sorted(list(self.counters['entities'])),
        }

class BasicPreprocessor:
    """Removes stop words, whitespaces, EOL"""

    def __init__(self, nlp):
        self.nlp = nlp
        self._stats = {}

    def __call__(self, whole_txt):
        language = self.nlp.Defaults
        # tokenizer = language.create_tokenizer(nlp)
        stop_words = language.stop_words

        self._stats['n_chars'] = len(whole_txt)

        tokens = self.nlp(whole_txt.replace('\t',' ')) # just tokenise: tokenizer() instead of nlp()
        
        self._stats['n_tokens_all'] = len(tokens)

        tokens = [t for t in tokens 
            if t.text not in stop_words 
            and not t.is_space 
            and t.text != '\n']

        self._stats['n_tokens_filtered'] = len(tokens)

        return tokens
    
    def stats(self):
        return self._stats

