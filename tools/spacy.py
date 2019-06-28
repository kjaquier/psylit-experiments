from collections import Counter, namedtuple
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
                rel = 'agent' if root.dep_ in ('nsubj', 'csubj', 'nsubjpass') else 'patient'
                for anc in root.ancestors:
                    anc_lemma = anc.lemma_
                    if anc_lemma == 'be':
                        rel = 'predicative'
                    elif anc_lemma in ('have', 'possess'):
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
            

