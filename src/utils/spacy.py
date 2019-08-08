import logging

from itertools import combinations

from nltk.corpus import wordnet as wn

import spacy.matcher as spmatch
from spacy.tokens import Token


logger = logging.getLogger(__name__)


def filter_spans(spans):
    """Filter a sequence of spans and remove duplicates or overlaps. Useful for
    creating named entities (where one token can only be part of one entity) or
    when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
    longest span is preferred over shorter spans.
    spans (iterable): The spans to filter.
    RETURNS (list): The filtered spans.
    """
    # From https://github.com/explosion/spaCy/pull/3686
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result

    
def fix_names(doc):
    """Spacy pipeline component for merging particles like 'Mr/Mrs' etc."""
    matcher = spmatch.Matcher(doc.vocab)
    matcher.add('name_parts', None, [{'DEP': {'IN': ('compound', 'prt', 'flat', 'poss')}, 'ENT_IOB': {'NOT': 'O'}, 'OP':'+'},
                                     {'ENT_IOB': {'NOT': 'O'}}])
    matches = matcher(doc)
    spans = filter_spans(doc[s:e] for _, s, e in matches)
    logger.debug('%s matches', len(spans))
    with doc.retokenize() as retokenizer:
        for span in spans:
            root = span.root
            attrs = {
                'DEP': root.dep,
                'POS': root.pos,
                'TAG': root.tag,
                'ENT_TYPE': root.ent_type,
                'ENT_IOB': root.ent_iob,
            }
            retokenizer.merge(span, attrs)
    return doc
            

class RemoveExtensionsMixin:

    def __init__(self, extensions=None, **kwargs):
        self.exts = []
        self.kwargs = kwargs
        for cls, attr_name, kw in (extensions or []):
            self.set_extension(cls, attr_name, **kwargs, **kw)

    def set_extension(self, cls, attr_name, **kwargs):
        # logger.debug(f"set extension {cls.__name__}._.{attr_name} {kwargs!r}")
        cls.set_extension(attr_name, **self.kwargs, **kwargs)
        self.exts.append((cls, attr_name, kwargs))

    def remove_extensions(self):
        for cls, attr_name, _ in self.exts:
            # logger.debug(f"remove extension {cls.__name__}._.{attr_name}")
            cls.remove_extension(attr_name)

    def get_extensions_remover_component(self):
        
        def component(doc):
            self.remove_extensions()
            return doc

        return component


class HypernymsExtractor(RemoveExtensionsMixin):
    name = 'hypernyms'
    
    def __init__(self, vocab, pattern=None, highest_level=1, force_ext=False):
        super().__init__()
        super().set_extension(Token, 'hypernyms', default=set(), force=force_ext)
        if pattern:
            self.matcher = spmatch.Matcher(vocab)
            self.matcher.add('hypernyms', None, pattern)
        else:
            self.matcher = None
        self.highest_level = highest_level
    
    def __call__(self, doc):
        if self.matcher:
            matches = self.matcher(doc)
            logger.debug("%s matches", len(matches))
            toks = (doc[start:end].root for _, start, end in matches)
        else:
            toks = iter(doc)
        
        for t in toks:
            
            hypers = {h for s in t._.wordnet.synsets() for h in s.hypernyms()}
            i = 0
            while len(hypers) > 1 and i < self.highest_level:
                hypers_pairs = combinations(hypers, 2)
                hypers = {h for h1, h2 in hypers_pairs for h in h1.lowest_common_hypernyms(h2)}
                i += 1

            t._.set('hypernyms', hypers)
            
        return doc


class HypernymMatcher:

    def __init__(self, targets, highest_level=0, highest_common_level=0):
        self.targets = {wn.synset(k) for k in targets}
        self.highest_level = highest_level
        self.highest_common_level = highest_common_level

    def match_token(self, token):
        targets = self.targets
        matched = set()
        
        hypers_all = hypers_common = {h for s in token._.wordnet.synsets() for h in s.hypernyms()}
        matched |= targets & hypers_all

        for _ in range(self.highest_level):
            hypers_all = {h for s in hypers_all for h in s.hypernyms()}
            if not hypers_all:
                break
            matched |= targets & hypers_all

        for _ in range(self.highest_common_level):
            hypers_pairs = combinations(hypers_common, 2)
            hypers_common = {h for h1, h2 in hypers_pairs for h in h1.lowest_common_hypernyms(h2)}
            if not hypers_common:
                break
            matched |= targets & hypers_common

        return matched
