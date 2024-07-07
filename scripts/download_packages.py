import nltk
import spacy

nlp = spacy.load("en_core_web_md")


def download_packages():
    for package in ['stopwords', 'punkt']:
        nltk.download(package)


if __name__ == "__main__":
    download_packages()
