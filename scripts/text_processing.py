from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


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
