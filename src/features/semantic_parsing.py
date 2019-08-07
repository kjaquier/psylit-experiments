from itertools import combinations, product
from collections import Counter, namedtuple, defaultdict

from nltk.corpus import wordnet as wn

import spacy.matcher as spmatch
from spacy.tokens import Doc, Span, Token

from utils import spacy as spacy_utils

AGENT_DEPS = ('nsubj', 'csubj', 'poss', 'expl')
PATIENT_DEPS = ('nsubjpass','csubjpass', 'obj','pobj','dobj','iobj','auxpass','nmod')
DEP_WHITELIST = (
                 'conj','compound','neg','poss',
                 'prep','amod','attr','acl','advcl',
                 'appos','aux','dislocated','obl',
                 'orphan', # connects agent/patient in the conj of a previous predicate
                 # clearly excluded:
                 # 'npadvmod', 'advmod', 'parataxis', 'xcomp'
                 )

class SemanticDepParser(spacy_utils.RemoveExtensionsMixin):
    """Propagates agent to ancestors"""

    name = 'semantic_dep'

    def __init__(self, force_ext=False, predicate_flag='has_lex'):
        super().__init__(force=force_ext)
        super().set_extension(Token, 'agents', default=set())
        super().set_extension(Token, 'patients', default=set())
        super().set_extension(Span, 'predicates', default=set())
        super().set_extension(Token, 'subsent_root', default=None)
        #super().set_extension(Doc, 'predicates', default=list())
        self.predicate_flag = predicate_flag

    def __call__(self, doc):
        #coref_tokens = {
        #    t.i for c in doc._.coref_clusters for m in c.mentions for t in m
        #}
        predicate_flag = self.predicate_flag
        dep_inside_subsentence = DEP_WHITELIST + PATIENT_DEPS
        
        #visited = defaultdict(lambda: set())
        #visited.update({t.i:{'Pr'} for t in doc if t._.get(predicate_flag)})
        
        cnt = Counter()
        clusters = doc._.coref_clusters
        patients = set()

        # find agents and patients
        for clust in clusters:
            ent = clust.main
            for mention in clust:
                mention_root = mention.root
                head = mention_root.head 
                is_agent = mention_root.dep_ in AGENT_DEPS
                #is_co_agent = mention_root.dep_ == 'conj' and head.dep_ == in AGENT_DEPS) # FIXME doesn't deal with "X but Y" etc.
                
                if is_agent:
                    #cnt['Ag'] += 1
                    # TODO also add ancestors through conj etc. 
                    head._.get('agents').add(ent)
                    mention_root._.set('subsent_root', head)

                    if head != mention_root:
                        pass
                        #cnt['Head'] += 1
                    #roots.add((head, ent))
                
                else: # then patient
                    patients.add(mention)

                    #cnt['Pat'] += 1

        # resolve predicate of all patients by looking at ancestors
        # until resolved or hit a subsentence root
        for patient in patients:
            patient_root = patient.root
            if patient_root.dep_ in dep_inside_subsentence: 
                for ances in patient_root.ancestors:
                    if ances._.get(predicate_flag):
                        ances._.patients.add(patient)
                        patient_root._.set('subsent_root', ances)
                        #cnt['+Pat'] += 1
                        break

                    if ances._.agents or ances.dep_ not in dep_inside_subsentence: # resolved or subsentence root
                        patient_root._.set('subsent_root', ances)
                        #cnt['^Pat Stub :('] += 1
                        #cnt['Stub :('] += 1
                        break
                else:
                    patient_root._.set('subsent_root', patient_root.sent.root)
            else:
                patient_root._.set('subsent_root', patient_root)
                pass
                #cnt['Pat Stub :('] += 1
                #cnt['Stub :('] += 1

        # resolve agents of all predicates by looking at ancestors
        # until resolved or hit a subsentence root
        for predicate in doc._.lex_matches: # TODO make it more generic
            predicate_agents = predicate._.agents
            #cnt['W'] += 1

            # resolved?
            if predicate_agents:
                #cnt['+Ag'] += 1
                #cnt['W+Head+Ag'] += 1
                predicate._.set('subsent_root', predicate)
                continue
            
            # not resolved: iter ancestors
            if predicate.dep_ in dep_inside_subsentence: 
                for ances in predicate.ancestors:
                    agents = ances._.agents
                    if agents:
                        #cnt['*Ag'] += 1
                        #cnt['+Ag'] += 1
                        predicate._.agents.update(agents) # propagate agents
                        predicate._.set('subsent_root', ances)
                        break
                    
                    if ances.dep_ not in dep_inside_subsentence:
                        #cnt['Stub :('] += 1
                        #cnt['W^ Stub :('] += 1
                        predicate._.set('subsent_root', ances)
                        break
                else:
                    #cnt['Root :('] += 1
                    predicate._.set('subsent_root', predicate.sent.root)
                    pass
            else:
                predicate._.set('subsent_root', predicate)
                pass
                #cnt['W Stub :('] += 1
                #cnt['Stub :('] += 1

            # selected_childs = [
            #         c for c in tok.children 
            #         #if c.dep_ in dep_inside_subsentence and  # some edges represent sub-sentences that we want to keep separate
            #         #   not c in roots   # if a node in the tree has an agent, we will propagate that one instead
            #     ]

        #print(f"[{self.__class__.__name__}] {Counter(' '.join(v) for v in visited.values())}")
        print(f"[{self.__class__.__name__}] {cnt}")

        return doc
