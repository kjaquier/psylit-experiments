from itertools import combinations
from collections import Counter, namedtuple, defaultdict

from nltk.corpus import wordnet as wn
from spacy.matcher import Matcher
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
    matcher = Matcher(doc.vocab)
    matcher.add('name_parts', None, [{'DEP': {'IN': ('compound','prt','flat')}, 'ENT_IOB': {'NOT': 'O'}, 'OP':'+'},
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
        print(f"[{self.__class__.__name__}] set extension {cls.__class__.__name__}._.{attr_name} {kwargs!r}")
        cls.set_extension(attr_name, **self.kwargs, **kwargs)
        self.exts.append((cls, attr_name, kwargs))

    def remove_extensions(self):
        for cls, attr_name, _ in self.exts:
            print(f"[{self.__class__.__name__}] remove extension {cls.__class__.__name__}._.{attr_name}")
            cls.remove_extension(attr_name)

    def get_extensions_remover_component(self):
        
        def component(doc):
            self.remove_extensions()
            return doc

        return component

class LexiconTagger(RemoveExtensionsMixin):
    
    name = 'lexicon'

    def __init__(self, vocab, lexicon, tag_attr='lex', flag_attr=None, force_ext=False):
        super().__init__(force=force_ext)
        self.tag_attr = tag_attr
        self.flag_attr = flag_attr or ('has_' + tag_attr)
        self.tags = lexicon.columns
        
        super().set_extension(Token, self.tag_attr, default=set())
        super().set_extension(Token, self.flag_attr, default=False)

        self.matcher = Matcher(vocab)
        for tag in self.tags:
            terms = lexicon.loc[lexicon[tag] > 0, tag].index # TODO deal with loadings
            terms = list(terms.unique())
            
            self.matcher.add(tag, None, [{'LOWER': {'IN': terms}}])
            self.matcher.add(tag, None, [{'LEMMA': {'IN': terms}}])
    
    def __call__(self, doc):
        matches = self.matcher(doc)
        tag_attr = self.tag_attr
        flag_attr = self.flag_attr
        print(f"[{self.__class__.__name__}] {len(matches)} matches")
        for matched_tag, start, end in matches:
            tag = matched_tag
            for tok in doc[start:end]:
                tok._.get(tag_attr).add(tag)
                tok._.set(flag_attr, True)
        
        return doc


class HypernymTagger(RemoveExtensionsMixin):
    
    def __init__(self, vocab, wndomains, tag_attr, force_ext=False):
        super().__init__(force=force_ext)
        self.tag_attr = tag_attr
        self.wndomains = wndomains
        super().set_extension(Token, tag_attr, default=None)
    
    def __call__(self, doc):
        domains = self.wndomains
        for tok in doc:
            synsets = tok._.wordnet.wordnet_synsets_for_domain(domains)
            
            if not synsets:
                continue
            
            categs = Counter(h for ps in synsets for h in ps.hypernyms())
            n_categs = len(categs.keys())
            most_likely_cat, _cat_cnt = categs.most_common(1)[0]
            tok._.set(self.tag_attr, most_likely_cat)

        return doc
        

class NegTagger(RemoveExtensionsMixin):
    name = 'negation'
    
    def __init__(self, vocab, force_ext=False):
        super().__init__(force=force_ext)
        super().set_extension(Token, 'negated', default=False)
        self.matcher = Matcher(vocab)
        self.matcher.add('neg', None, [{'DEP': 'neg'}])
        
    def __call__(self, doc):
        matches = self.matcher(doc)
        print(f"[{self.__class__.__name__}] {len(matches)} matches")
        for _, start, end in matches:
            head = doc[start:end].root.head
            if head:
                head._.set('negated', True)

        return doc


AGENT_DEPS = {'nsubj', 'csubj', 'nsubjpass', 'poss'}
PREDICATIVE_LEMMAS = ('be',)
POSSESSIVE_LEMMAS = ('have', 'possess')


class SemanticDepParser(RemoveExtensionsMixin):

    name = 'semantic_dep'

    def __init__(self, force_ext=False):
        super().__init__(force=force_ext)
        super().set_extension(Token, 'sem_deps', default=list())

    def __call__(self, doc):
        clusters = doc._.coref_clusters
        print(f"[{self.__class__.__name__}] {len(clusters)} clusters")
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
        print(f"[{self.__class__.__name__}]", ', '.join(f"{k}:{v}" for k,v in cnt.items()))
        return doc


class PredicateParser(RemoveExtensionsMixin):

    name = 'predicates'

    def __init__(self, vocab, pattern=[{'_': {'has_lex': True}}], force_ext=False):
        super().__init__(force=force_ext)
        assert Token.has_extension('sem_deps'), "Need extension 'sem_deps'!"
        self.matcher = Matcher(vocab)
        self.matcher.add('predicate', None, pattern)

    def __call__(self, doc):
        matches = self.matcher(doc)
        print(f"[{self.__class__.__name__}] {len(matches)} matches")
        cnt = Counter()
        for _, start, end in matches:
            heir = doc[start:end].root
            rels = {(rel, clust.i): clust for rel, clust in heir._.sem_deps}
            for anc in heir.ancestors:
                for rel, target in anc._.sem_deps:
                    rels[rel, target.i] = target
                    cnt[rel] += 1
            heir._.sem_deps = [(rel, clust) for (rel, _), clust in rels.items()]
        print(f"[{self.__class__.__name__}]", ', '.join(f"{k}:{v}" for k,v in cnt.items()))
        return doc
            

class HypernymsExtractor(RemoveExtensionsMixin):
    name = 'hypernyms'
    
    def __init__(self, vocab, pattern=None, highest_level=1, force_ext=False):
        super().__init__()
        super().set_extension(Token, 'hypernyms', default=set(), force=force_ext)
        if pattern:
            self.matcher = Matcher(vocab)
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
            toks= iter(doc)
        
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
            self.matcher = Matcher(vocab)
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
            toks= iter(doc)
        
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
    

