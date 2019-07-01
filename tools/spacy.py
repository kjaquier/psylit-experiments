from itertools import combinations
from collections import Counter, namedtuple, defaultdict

from nltk.corpus import wordnet as wn
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token

class LexiconTagger:
    
    name = 'lexicon'

    def __init__(self, vocab, lexicon, tag_attr='lex', flag_attr=None, force_ext=True):
        self.tag_attr = tag_attr
        self.flag_attr = flag_attr or ('has_' + tag_attr)
        self.tags = lexicon.columns
        
        Token.set_extension(self.tag_attr, default=set(), force=force_ext)
        Token.set_extension(self.flag_attr, default=False, force=force_ext)

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


class HypernymTagger:
    
    def __init__(self, vocab, wndomains, tag_attr, force_ext=True):
        self.tag_attr = tag_attr
        self.wndomains = wndomains
        Token.set_extension(tag_attr, default=None, force=force_ext)
    
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
        

class NegTagger:
    name = 'negation'
    
    def __init__(self, vocab, force_ext=True):
        Token.set_extension('negated', default=False, force=force_ext)
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


class CorefParser__old:
    
    name = 'corefparse'
    relations = ['agent', 'patient', 'predicative']
    
    def __init__(self, vocab, force_ext=True):
        for r in self.relations:
            Token.set_extension(r, default=None, force=force_ext)
        self.matcher = Matcher(vocab)
        self.matcher.add('agent', None, [{'_': {'in_coref': True}}])
        #self.matcher.add('agent', None, [{'_': {'in_coref': True}, 'DEP': {'IN', ['nsubj', 'csubj']}}])
        #self.matcher.add('patient', None, [{'_': {'in_coref': True}, 'DEP': {'NOT_IN', ['nsubj', 'csubj']}}])
        
    def __call__(self, doc):
        i=0
        
        matches = self.matcher(doc)
        for relation, start, end_ in matches:
            #assert end_ - start == 1, f"doc[{start}:{end_}]={doc[start:end_].text!r}"
            tok = doc[start]
            
            is_subj = tok.dep_ in ('nsubj', 'csubj', 'nsubjpass')
            #for is_subj, tok in ((, t) for t in doc if t._.in_coref):
            relation = 'agent' if is_subj else 'patient'
            target = tok._.coref_clusters[0].main
                
            for anc in tok.ancestors:
                if anc._.get(relation):
                    i+=1
                anc._.set(relation, target)
            
            #for desc in tok.head.subtree: # TODO excluding tok.conjuncts and tok
            #    desc._.set('predicative', clust.main)
            
            #root = tok.sent.root
            #if anc.lemma_ == 'be':
            #    candidate_roots = [t for t in root.children if t.dep_ in ('acomp','dcomp')]
            #    root = candidate_roots[0] if candidate_roots else root
            #    relation = 'predicative'
            #clust = tok._.coref_clusters[0]
            #root._.set(relation, clust.main)
        
        if i:
            print(f'[WARNING] {i} conflicts in coreference resolution!', )
        
        return doc
    
    #def __del__(self):
    #    for r in self.relations:
    #        Token.remove_extension(r)

AGENT_DEPS = {'nsubj', 'csubj', 'nsubjpass', 'poss'}
PREDICATIVE_LEMMAS = ('be',)
POSSESSIVE_LEMMAS = ('have', 'possess')

class SemanticDepParser:

    name = 'semantic_dep'

    def __init__(self, force_ext=True):
        Token.set_extension('sem_deps', default=list(), force=force_ext)

    def __call__(self, doc):
        clusters = doc._.coref_clusters
        print(f"[{self.__class__.__name__}] {len(clusters)} clusters")
        for clust in clusters:
            main = clust.main
            for mention in clust:
                root = mention.root
                rel = 'agent' if root.dep_ in AGENT_DEPS else 'patient'
                for anc in root.ancestors:
                    anc_lemma = anc.lemma_
                    if anc_lemma == PREDICATIVE_LEMMAS:
                        rel = 'predicative'
                    elif anc_lemma in POSSESSIVE_LEMMAS:
                        rel = 'possessive'
                    
                    anc._.sem_deps.append((rel, main))
        return doc


class PredicateParser:

    name = 'predicates'

    def __init__(self, vocab, pattern=[{'_': {'has_lex': True}}]):
        assert Token.has_extension('sem_deps'), "Need extension 'sem_deps'!"
        self.matcher = Matcher(vocab)
        self.matcher.add('predicate', None, pattern)

    def __call__(self, doc):
        matches = self.matcher(doc)
        print(f"[{self.__class__.__name__}] {len(matches)} matches")
        for _, start, end in matches:
            heir = doc[start:end].root
            heir_rels = {}
            for anc in heir.ancestors:
                for rel, target in anc._.sem_deps:
                    if rel not in heir_rels:
                        heir_rels[rel] = target
        return doc
            

class HypernymsExtractor:
    name = 'hypernyms'
    
    def __init__(self, vocab, pattern=None, highest_level=1, force_ext=True):
        Token.set_extension('hypernyms', default=set(), force=force_ext)
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
    targets = {wn.synset(k) for k in targets}

    def match(token):
        matched = set()
        
        hypers_all = hypers_common = {h for s in token._.wordnet.synsets() for h in s.hypernyms()}
        matched |= targets & hypers_all
        for i in range(highest_level):
            hypers_all = {h for s in hypers_all for h in s.hypernyms()}
            if not hypers_all:
                print("i=",i)
                break
            matched |= targets & hypers_all

        for i in range(highest_common_level):
            hypers_pairs = combinations(hypers_common, 2)
            hypers_common = {h for h1, h2 in hypers_pairs for h in h1.lowest_common_hypernyms(h2)}
            if not hypers_common:
                print("j=",i)
                break
            matched |= targets & hypers_common

        return matched
    
    return match


class HypernymMatcher:
    name = 'hypernym_match'
    
    def __init__(self, vocab, synset2tag, 
                 pattern=None, highest_level=0, 
                 attr_name='hypernym_match', 
                 highest_common_level=0, force_ext=True):
        Token.set_extension(attr_name, default=set(), force=force_ext)
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


class EntityTagger:
    name = 'entity_tag'

    def __init__(self, collector, attr_name='ent_tags', force_ext=True):
        """
        collector: Function[Span] -> Set[Any]
        """
        Doc.set_extension(attr_name, default=None, force=force_ext)
        Span.set_extension(attr_name, default=None, force=force_ext)
        self.collector = collector
        self.attr_name = attr_name
    
    def __call__(self, doc):
        collector = self.collector
        alltags = defaultdict(set)
        for clust in doc._.coref_clusters:
            clust_id = clust.i
            collected = alltags[clust_id]
            for mention in clust:
                tags = collector(mention)
                mention._.set(self.attr_name, tags)
                collected.update(tags)
        
        doc._.set(self.attr_name, alltags)
        return doc
