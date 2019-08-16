from functools import partial
import logging

from spacy.matcher import Matcher

from utils import spacy as spacy_utils


logger = logging.getLogger(__name__)


PERSON, ENV, UNK, NARR, READ, NAN = ['PERSON', 'ENVIRONMENT', 'UNKNOWN', 'NARRATOR', 'READER', None]
    

PERSON_ENT_TYPE = 'PERSON'
ENV_ENT_TYPES = {'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT'}


EXCEPTIONS = {
    ENV: {'it', 'this', 'that', 'its', 'itself', 'something'},
    UNK: {'they', 'them', 'themselves', 'their', 'these', 'those'},
    NARR: {'i', 'me', 'myself', 'my', 'mine',
           'we', 'us', 'ourselves', 'our', 'ours'},
    READ: {'you', 'yourself', 'your', 'yours'},
    PERSON: {'he', 'him', 'himself', 'his',
             'she', 'her', 'herself', 'hers',
             'man', 'boy', 'sir',
             'woman', 'girl', 'madam', 'miss', 'lord', 'lady'},
}
EXCEPTIONS = {w: k for k, v in EXCEPTIONS.items() for w in v}


HYPERNYM_MAP = {'person.n.01': 'person', 
                'female.n.02':'female', 
                'woman.n.01':'female',
                'man.n.01':'male',
                'male.n.02': 'male', 
                'entity.n.01': 'entity'}

EntityTypeHypernymMatcher = partial(spacy_utils.HypernymMatcher, HYPERNYM_MAP.keys(),
                                    highest_level=10,
                                    highest_common_level=0)


def entity_classifier(vocab):
    exceptions_matcher = Matcher(vocab)
    exceptions_matcher.add('detpron', None, [{'POS': {'IN': ('DET', 'PRON')}}])#, '_': {'in_coref': False}}])

    def iter_entities(doc):
        coref_tokens = {
            t.i 
            for c in (doc._.coref_clusters or []) 
            for m in (c.mentions or [])
            for t in m
        }

        matches = exceptions_matcher(doc)

        for _, start, end in matches:
            mention = doc[start:end]
            m_root = mention.root
            if m_root.i in coref_tokens:
                continue
            m_root_text = m_root.text.lower()
            ent_class = EXCEPTIONS.get(m_root_text, UNK)

            sent = m_root.sent
            yield {
                'i': m_root.i,
                't0': sent.start,
                't1': sent.end,
                'entity_i': None,
                'entity_root': ent_class,
                #'entity_pos': None,
                #'entity_tag': None,
                #'entity_text': None,
                'mention_text': mention.text,
                'mention_root': m_root_text,
                'mention_pos': m_root.pos_,
                'mention_tag': m_root.tag_,
                'categ': ent_class.lower() if ent_class else None,
            }

        for cluster in doc._.coref_clusters:
            e = cluster.main
            e_root = e.root
            e_i = e_root.i # cluster.i
            e_root_text = e_root.text.strip().lower()
            # e_root_text = EXCEPTIONS.get(e_root_text, '') or e_root_text
            e_pos = e.root.pos_
            e_tag = e.root.tag_
            e_txt = e.text
            for mention in cluster.mentions:
                c_root = e_root or mention.root
                m_root = mention.root
                m_root_text = m_root.text.lower()
                sent = m_root.sent
                #c_root = cluster.main.root if cluster else mention.root
                if c_root.pos_ == 'PROPN':  # resolved to an entity
                    if c_root.ent_type_ == PERSON_ENT_TYPE:
                        ent_class = PERSON
                    elif c_root.ent_type_ in ENV_ENT_TYPES:
                        ent_class = ENV
                    else: # dates, quantities etc.: ignore those
                        ent_class = EXCEPTIONS.get(m_root_text, NAN)
                elif c_root.pos_ == 'NOUN':  # resolved to a noun
                    hypernyms = {HYPERNYM_MAP[m.name()] for m in c_root._.hypernyms}
                    if 'person' in hypernyms:
                        ent_class = PERSON
                    else:
                        ent_class = ENV
                elif c_root.pos_ in {'DET', 'PRON'}: # unresolved
                    c_root_text = c_root.text.lower()
                    
                    ret = EXCEPTIONS.get(c_root_text, None)
                    ret = ret or EXCEPTIONS.get(m_root_text, None)
                    
                    # unlikely a det/pron doesn't refer to a person (except exceptions)
                    ent_class = ret or PERSON

                else: # verbs, adj etc... most likely irrelevant (or unknown?)
                    ent_class = UNK
                
                yield {
                    'i': mention.start,
                    't0': sent.start,
                    't1': sent.end,
                    'entity_i': e_i,
                    #'entity': e_selected,
                    'entity_root': e_root_text,
                    'entity_pos': e_pos,
                    'entity_tag': e_tag,
                    'entity_text': e_txt,
                    'mention_text': mention.text,
                    'mention_root': m_root_text,
                    'mention_pos': m_root.pos_,
                    'mention_tag': m_root.tag_,
                    'categ': ent_class.lower() if ent_class else None,
                }
    return iter_entities
    #Doc.set_extension(Doc, doc_attr, method=iter_entities)
