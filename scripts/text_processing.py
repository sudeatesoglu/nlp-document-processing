from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from handle_document import read_document
import string

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def process_document(doc_type, doc_1, doc_2):
    document_1 = read_document(doc_type, doc_1)
    document_1 = preprocess_text(document_1)
    document_2 = read_document(doc_type, doc_2)
    document_2 = preprocess_text(document_2)
    return document_1, document_2


def remove_punctuation(text):
    punctuation = list(string.punctuation)
    return ''.join([char if char not in punctuation else ' ' for char in text])


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    return ' '.join([word for word in word_tokens if word.lower() not in stop_words])


def stem_text(text):
    word_tokens = word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in word_tokens])


def lemmatize_text(text):
    word_tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in word_tokens])
