from collections import Counter
import logging

import spacy.matcher as spmatch
from spacy.tokens import Doc, Token

from utils import spacy as spacy_utils


class LexiconTagger(spacy_utils.RemoveExtensionsMixin):   
    name = 'lexicon'

    def __init__(self, nlp, lexicon, tag_attr='lex', flag_attr=None,
                 doc_attr='lex_matches', force_ext=False):
        super().__init__(force=force_ext)
        self.tag_attr = tag_attr
        self.flag_attr = flag_attr or ('has_' + tag_attr)
        self.ances_flag_attr = 'child_' + self.flag_attr
        self.doc_attr = doc_attr
        self.tags = lexicon.columns
        
        super().set_extension(Token, self.tag_attr, default=set())
        super().set_extension(Token, self.flag_attr, default=False)
        super().set_extension(Token, self.ances_flag_attr, default=False)
        super().set_extension(Doc, self.doc_attr, default=list())

        self.matcher = spmatch.Matcher(nlp.vocab)
        for tag in self.tags:
            terms = lexicon.loc[lexicon[tag] > 0, tag].index # TODO deal with loadings
            terms = list(terms.unique())
            self.matcher.add(tag, None, [{'LOWER': {'IN': terms}}], [{'LEMMA': {'IN': terms}}])

    def __call__(self, doc):
        matches = self.matcher(doc)
        tag_attr = self.tag_attr
        flag_attr = self.flag_attr
        doc_attr = self.doc_attr
        ances_flag_attr = self.ances_flag_attr

        logging.info(f"[{self.__class__.__name__}] {len(matches)} matches")

        tok_list = doc._.get(doc_attr)

        tagged = set()

        i = 0
        j = 0
        for matched_tag, start, end in matches:
            span = doc[start:end]
            for tok in span:
                tok._.get(tag_attr).add(matched_tag)
                tok._.set(flag_attr, True)
                tok._.set(ances_flag_attr, True)
                i += 1
                
                if tok not in tagged:    
                    tok_list.append(tok)
                    tagged.add(tok)
                
            for ances in span.root.ancestors:
                if ances._.get(ances_flag_attr):
                    break
                ances._.set(ances_flag_attr, True)
                j += 1

        logging.debug(f"[{self.__class__.__name__}] {i} tags assigned on {len(tagged)}"
                      f" tokens, {j} ancestors flagged")
        return doc


class FastLexiconTagger(spacy_utils.RemoveExtensionsMixin):
    
    name = 'lexicon'

    def __init__(self, nlp, lexicon, tag_attr='lex', flag_attr=None, doc_attr='lex_matches', force_ext=False):
        super().__init__(force=force_ext)
        self.tag_attr = tag_attr
        self.flag_attr = flag_attr or ('has_' + tag_attr)
        self.ances_flag_attr = 'child_' + self.flag_attr
        self.doc_attr = doc_attr
        self.tags = lexicon.columns
        
        super().set_extension(Token, self.tag_attr, default=set())
        super().set_extension(Token, self.flag_attr, default=False)
        super().set_extension(Token, self.ances_flag_attr, default=False)
        super().set_extension(Doc, self.doc_attr, default=list())

        self.matcher = spmatch.PhraseMatcher(nlp.vocab, attr='LOWER', validate=True) 
        # FIXME add match on LEMMA, for now doesn't match anything because of https://github.com/explosion/spaCy/commit/d59b2e8a0c595498d7585b23ebb461ce82719809
        # fixed on spacy 2.1.4+, need to update dependencies
        
        for tag in self.tags:
            terms = lexicon.loc[lexicon[tag] > 0, tag].index # TODO deal with loadings
            terms = [nlp.make_doc(t) for t in terms.unique()]

            self.matcher.add(tag, None, *terms)
            
    def __call__(self, doc):
        matches = self.matcher(doc)
        tag_attr = self.tag_attr
        flag_attr = self.flag_attr
        doc_attr = self.doc_attr
        ances_flag_attr = self.ances_flag_attr

        logging.info(f"[{self.__class__.__name__}] {len(matches)} matches")

        i = 0
        for matched_tag, start, end in matches:
            span = doc[start:end]
            for tok in span:
                tok._.get(tag_attr).add(matched_tag)
                tok._.set(flag_attr, True)
                tok._.set(ances_flag_attr, True)
                doc._.get(doc_attr).append(tok)
            i += 1
            for ances in span.root.ancestors:
                if ances._.get(ances_flag_attr):
                    break
                ances._.set(ances_flag_attr, True)
                i += 1
        
        logging.info(f"[{self.__class__.__name__}] {i} tags assigned")

        return doc


class HypernymTagger(spacy_utils.RemoveExtensionsMixin):
    
    def __init__(self, wndomains, tag_attr, force_ext=False):
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
            #n_categs = len(categs.keys())
            most_likely_cat, _cat_cnt = categs.most_common(1)[0]
            tok._.set(self.tag_attr, most_likely_cat)

        return doc
        

class NegTagger(spacy_utils.RemoveExtensionsMixin):
    name = 'negation'
    
    def __init__(self, vocab, force_ext=False):
        super().__init__(force=force_ext)
        super().set_extension(Token, 'negated', default=False)
        self.matcher = spmatch.Matcher(vocab)
        self.matcher.add('neg', None, [{'DEP': 'neg'}])
        
    def __call__(self, doc):
        matches = self.matcher(doc)
        print(f"[{self.__class__.__name__}] {len(matches)} matches")
        for _, start, end in matches:
            head = doc[start:end].root.head
            if head:
                head._.set('negated', True)

        return doc
