from nltk.tokenize import word_tokenize, sent_tokenize


def tokenize_document(document):
    sent_tokens = sent_tokenize(document)
    word_tokens = word_tokenize(sent_tokens)
    tokens = set(word_tokens)
    return tokens
