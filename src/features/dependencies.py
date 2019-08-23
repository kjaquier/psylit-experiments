
AGENT_DEPS = ('nsubj', 'csubj', 'poss', 'expl')
PATIENT_DEPS = ('nsubjpass', 'csubjpass', 'obj', 'pobj',
                'dobj', 'iobj', 'auxpass', 'nmod')
DEP_WHITELIST = (
    'conj', 'compound', 'neg', 'poss',
    'prep', 'amod', 'attr', 'acl', 'advcl',
    'appos', 'aux', 'dislocated', 'obl',
    'orphan',  # connects agent/patient in the conj of a previous predicate
    # clearly excluded:
    # 'npadvmod', 'advmod', 'parataxis', 'xcomp'
)
DEP_INSIDE_SUBSENTENCE = DEP_WHITELIST + PATIENT_DEPS
