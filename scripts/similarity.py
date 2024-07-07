import spacy
import numpy as np
from text_processing import preprocess_text

nlp = spacy.load("en_core_web_md")


def check_doc_similarity(doc_type, doc_1, doc_2):
    """
    Checks the similarity between two documents after preprocessing.
    :param doc_type: (str) the type of the documents ('PDF' or 'TXT').
    :param doc_1: (str) path to the first document.
    :param doc_2: (str) path to the second documents.
    :return: (str) similarity score of two documents.
    """
    if doc_type == "PDF":
        with open(doc_1, 'rb') as file:
            data_1 = file.read()
        with open(doc_2, 'rb') as file:
            data_2 = file.read()
    elif doc_type == "TXT":
        with open(doc_1, 'r') as file:
            data_1 = file.read()
        with open(doc_2, 'r') as file:
            data_2 = file.read()

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


def check_categorical_similarity(category, documents):
    document_token = [nlp(text) for text in documents]
    category_token = nlp(category)
    similarities = []
    for i, doc in enumerate(document_token):
        similarity = round(doc.similarity(category_token), 2)
        similarities.append((f"Document {i+1}", similarity))
    return similarities
