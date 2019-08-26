from spacy.tokens import Span, Token

from utils import spacy as spacy_utils

from .dependencies import AGENT_DEPS, DEP_INSIDE_SUBSENTENCE


class SemanticDepParser(spacy_utils.RemoveExtensionsMixin):
    """Propagates agent to ancestors"""

    name = 'semantic_dep'

    def __init__(self, force_ext=False, predicate_flag='has_lex', predicate_attr='lex_matches'):
        super().__init__(force=force_ext)
        super().set_extension(Token, 'agents', default=set())
        super().set_extension(Token, 'patients', default=set())
        super().set_extension(Span, 'predicates', default=set())
        super().set_extension(Token, 'subsent_root', default=None)
        
        self.predicate_flag = predicate_flag
        self.predicate_attr = predicate_attr

    def __call__(self, doc):
        predicate_flag = self.predicate_flag
        predicate_attr = self.predicate_attr
        
        clusters = doc._.coref_clusters
        patients = set()

        # find agents and patients, attach agent to its head
        for clust in clusters:
            ent = clust.main
            for mention in clust:
                mention_root = mention.root
                head = mention_root.head
                is_agent = mention_root.dep_ in AGENT_DEPS

                # FIXME doesn't deal with "X but Y" etc.
                #is_co_agent = mention_root.dep_ == 'conj' and head.dep_ == in AGENT_DEPS)

                if is_agent:
                    # TODO also add ancestors through conj etc.
                    head._.get('agents').add(ent)
                    mention_root._.set('subsent_root', head)

                    #if head != mention_root:
                    #    pass

                else: # then patient
                    patients.add(mention)

        # resolve predicate of all patients by looking at ancestors
        # until resolved or hit a subsentence root
        for patient in patients:
            patient_root = patient.root
            if patient_root.dep_ in DEP_INSIDE_SUBSENTENCE:
                for ances in patient_root.ancestors:
                    if ances._.get(predicate_flag):
                        ances._.patients.add(patient)
                        patient_root._.set('subsent_root', ances)
                        break

                    if ances._.agents or ances.dep_ not in DEP_INSIDE_SUBSENTENCE: # resolved or subsentence root
                        patient_root._.set('subsent_root', ances)
                        break
                else:
                    patient_root._.set('subsent_root', patient_root.sent.root)
            else:
                patient_root._.set('subsent_root', patient_root)

        # resolve agents of all predicates by looking at ancestors
        # until resolved or hit a subsentence root
        for predicate in doc._.get(predicate_attr):
            predicate_agents = predicate._.agents

            # resolved?
            if predicate_agents:
                predicate._.set('subsent_root', predicate)
                continue

            # not resolved: iter ancestors
            if predicate.dep_ in DEP_INSIDE_SUBSENTENCE:
                for ances in predicate.ancestors:
                    agents = ances._.agents
                    if agents:
                        predicate._.agents.update(agents) # propagate agents
                        predicate._.set('subsent_root', ances)
                        break

                    if ances.dep_ not in DEP_INSIDE_SUBSENTENCE:
                        predicate._.set('subsent_root', ances)
                        break
                else:
                    predicate._.set('subsent_root', predicate.sent.root)
            else:
                predicate._.set('subsent_root', predicate)

        return doc
