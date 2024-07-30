import spacy
from spacy import displacy
from handle_document import read_document
from text_processing import preprocess_text

nlp = spacy.load("en_core_web_md")


def entity_recognize(doc_type, doc):
    document = read_document(doc_type, doc)
    document_token = nlp(document)
    entities = [(entity.text, entity.label_) for entity in document_token.ents]
    return entities


def display_entity_recognize(doc_type, doc):
    document = read_document(doc_type, doc)
    document_token = nlp(document)
    ner = displacy.render(document_token, style="ent", jupyter=True)
    return ner


def ner_interface(doc_type, doc):
    entities = entity_recognize(doc_type, doc)
    entity_display = display_entity_recognize(doc_type, doc)
    return entities, entity_display
