import spacy
import numpy as np
from handle_document import read_document
from text_processing import preprocess_text

nlp = spacy.load("en_core_web_md")


def check_doc_similarity(doc_type, doc_1, doc_2):
    data_1 = read_document(doc_type, doc_1)
    data_2 = read_document(doc_type, doc_2)

    data_1 = preprocess_text(data_1)
    data_2 = preprocess_text(data_2)

    doc_1_nlp = nlp(data_1)
    doc_2_nlp = nlp(data_2)
    similarity = round(doc_1_nlp.similarity(doc_2_nlp), 2)

    return "Similarity score of documents: ", similarity


def catch_related_words(word):
    word_vector = nlp.vocab.vectors[nlp.vocab.strings[word]]
    most_similar_words = nlp.vocab.vectors.most_similar(np.asarray([word_vector]), n=50)
    similar_words_list = [nlp.vocab.strings[w] for w in most_similar_words[0][0]]
    return similar_words_list


def check_categorical_similarity(category, doc_type, doc):
    document = read_document(doc_type, doc)
    document = preprocess_text(document)
    category = preprocess_text(category)
    category_token = nlp(category)
    document_token = nlp(document)
    similarity = round(document_token.similarity(category_token), 2)
    return f"Similarity according to category: {similarity}"
