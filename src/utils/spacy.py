import logging

from itertools import combinations
from collections import Counter, defaultdict

from nltk.corpus import wordnet as wn

import spacy.matcher as spmatch
from spacy.tokens import Doc, Span, Token


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
    print(f'[fix_names] {len(spans)} matches')
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
        # logging.debug(f"[{self.__class__.__name__}] set extension {cls.__name__}._.{attr_name} {kwargs!r}")
        cls.set_extension(attr_name, **self.kwargs, **kwargs)
        self.exts.append((cls, attr_name, kwargs))

    def remove_extensions(self):
        for cls, attr_name, _ in self.exts:
            # logging.debug(f"[{self.__class__.__name__}] remove extension {cls.__name__}._.{attr_name}")
            cls.remove_extension(attr_name)

    def get_extensions_remover_component(self):
        
        def component(doc):
            self.remove_extensions()
            return doc

        return component


class FlagLookup(RemoveExtensionsMixin):

    def __init__(self, cls, attr, doc_attr=None):
        super().__init__()
        self.cls = cls
        self.attr = attr
        self.doc_attr = doc_attr or f"lookup_{cls.__name__}_{attr}"
        assert cls.has_extension(attr), "Extension {attr!r} not registered in {cls.__name__!r}"
        super().set_extension(Doc, self.doc_attr, default=set())

    def _get_doc_attr(self):
        """Doesn't work on Doc"""
        return Doc._.get(self.doc_attr)

    def __call__(self, doc):
        raise NotImplementedError


class CorefLookup(FlagLookup):

    def __init__(self):
        attr = 'coref_tokens'
        super().__init__(Token, 'in_coref', doc_attr=attr)
    
    def __call__(self, doc):
        #indices = {
        #    t.i for c in doc._.coref_clusters for m in c.mentions for t in m
        #}
        #print(f"[{self.__class__.__name__}] {len(indices)} to be cached")
        #self._get_doc_attr().update(indices)
        #Doc._.get('coref_tokens').update(indices)
        return doc


AGENT_DEPS = {'nsubj', 'csubj', 'nsubjpass', 'poss'}
PREDICATIVE_LEMMAS = ('be',)
POSSESSIVE_LEMMAS = ('have', 'possess')

def ilen(gen):
    return sum(1 for _ in gen)


def union(sets):
    u = set()
    for s in sets:
        u |= s
    return u




class PredicateParser(RemoveExtensionsMixin):
    name = 'predicates'

    def __init__(self, vocab, pattern=[{'_': {'has_lex': True}}], force_ext=False):
        super().__init__(force=force_ext)
        super().set_extension(Token, 'sem_deps', default=list())
        self.matcher = spmatch.Matcher(vocab)
        self.matcher.add('predicate', None, pattern)

    def __call__(self, doc):
        
        clusters = doc._.coref_clusters
        logging.debug(f"[{self.__class__.__name__}] {len(clusters)} clusters")
        cnt = Counter()
        for clust in clusters:
            for mention in clust:
                root = mention.root
                rel = 'agent' if root.dep_ in AGENT_DEPS else 'patient'
                for anc in root.ancestors:
                    anc_lemma = anc.lemma_
                    if anc_lemma in PREDICATIVE_LEMMAS:
                        rel = 'predicative'
                    elif anc_lemma in POSSESSIVE_LEMMAS or doc.vocab.morphology.tag_map[anc.tag_].get('Poss', '') == 'yes':
                        rel = 'possessive'

                    anc._.sem_deps.append((rel, clust))
                    cnt[rel] += 1
        logging.debug(f"[{self.__class__.__name__}]", ', '.join(f"{k}:{v}" for k, v in cnt.items()))

        matches = self.matcher(doc)
        logging.debug(f"[{self.__class__.__name__}] {len(matches)} matches")
        cnt = Counter()
        for _, start, end in matches:
            heir = doc[start:end].root
            rels = {(rel, clust.i): clust for rel, clust in heir._.sem_deps}
            for anc in heir.ancestors:
                for rel, target in anc._.sem_deps:
                    rels[rel, target.i] = target
                    cnt[rel] += 1
            heir._.sem_deps = [(rel, clust) for (rel, _), clust in rels.items()]
        logging.debug(f"[{self.__class__.__name__}]", ', '.join(f"{k}:{v}" for k, v in cnt.items()))
        return doc
            

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
            print(f"[{self.__class__.__name__}] {len(matches)} matches")
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


def tok_hypernyms_matcher(targets, highest_level=0, highest_common_level=0):
    print("[WARNING DEPRECATED] MOVED TO processing.entities")
    targets = {wn.synset(k) for k in targets}

    def matcher(token):
        matched = set()
        
        hypers_all = hypers_common = {h for s in token._.wordnet.synsets() for h in s.hypernyms()}
        matched |= targets & hypers_all
        for _ in range(highest_level):
            hypers_all = {h for s in hypers_all for h in s.hypernyms()}
            if not hypers_all:
                break
            matched |= targets & hypers_all

        for _ in range(highest_common_level):
            hypers_pairs = combinations(hypers_common, 2)
            hypers_common = {h for h1, h2 in hypers_pairs for h in h1.lowest_common_hypernyms(h2)}
            if not hypers_common:
                break
            matched |= targets & hypers_common

        return matched
    
    return matcher


class HypernymMatcher(RemoveExtensionsMixin):
    name = 'hypernym_match'
    
    def __init__(self, vocab, synset2tag, 
                 pattern=None, highest_level=0, 
                 attr_name='hypernym_match', 
                 highest_common_level=0, force_ext=False):
        super().__init__(force=force_ext)
        super().set_extension(Token, attr_name, default=set())
        self.synset2tag = {wn.synset(k):v for k, v in synset2tag.items()}
        if pattern:
            self.matcher = spmatch.Matcher(vocab)
            self.matcher.add(attr_name, None, pattern)
        else:
            self.matcher = None
        
        self.attr_name = attr_name
        self.highest_level = highest_level
        self.highest_common_level = highest_common_level
    
    def __call__(self, doc):
        if self.matcher:
            matches = self.matcher(doc)
            print(f"[{self.__class__.__name__}] {len(matches)} matches")
            toks = (doc[start:end].root for _, start, end in matches)
        else:
            toks = iter(doc)
        
        attr_name = self.attr_name
        
        mention_matcher = tok_hypernyms_matcher(self.synset2tag.keys(),
                                                self.highest_level,
                                                self.highest_common_level)

        for t in toks:
            matched = mention_matcher(t)
            
            if matched:
                print(f"[{self.__class__.__name__}] matched for {t.text!r}: {matched} ")
            tags = {self.synset2tag[x] for x in matched}
            t._.set(attr_name, tags)
            
        return doc


class EntityMultiTagger(RemoveExtensionsMixin):
    name = 'entity_tags'

    def __init__(self, tagger, container=set, attr_name='ent_tags', force_ext=False):
        """
        tagger: Function[Cluster, Span] -> Iterable[Hashable]
        """
        print("[WARNING DEPRECATED] MOVED TO processing.entities")
        super().__init__(force=force_ext)
        super().set_extension(Doc, attr_name, default=None)
        super().set_extension(Span, attr_name, default=None)
        self.tagger = tagger
        self.container = container
        self.attr_name = attr_name
    
    def __call__(self, doc):
        tagger = self.tagger
        attr_name = self.attr_name
        alltags = defaultdict(self.container)
        for clust in doc._.coref_clusters:
            clust_id = clust.i
            collected = alltags[clust_id]
            for mention in clust:
                tags = tagger(clust, mention)
                mention._.set(attr_name, tags)
                collected.update(tags)
        
        doc._.set(attr_name, alltags)
        return doc


class EntityTagger(RemoveExtensionsMixin):
    name = 'entity_tag'

    def __init__(self, tagger, attr_name='ent_tag', force_ext=False):
        """
        tagger: Function[Cluster, Span] -> Hashable
        """
        print("[WARNING DEPRECATED] MOVED TO processing.entities")
        super().__init__(force=force_ext)
        super().set_extension(Span, attr_name, default=None)
        self.tagger = tagger
        self.attr_name = attr_name
    
    def __call__(self, doc):
        tagger = self.tagger
        attr_name = self.attr_name
        for clust in doc._.coref_clusters:
            for mention in clust:
                tag = tagger(clust, mention)
                mention._.set(attr_name, tag)
        return doc
