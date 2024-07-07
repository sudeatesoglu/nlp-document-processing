import spacy

nlp = spacy.load("en_core_web_md")


def entity_recognize(document):
    document_token = nlp(document)
    for entity in document_token.ents:
        return entity.text, entity.label_
