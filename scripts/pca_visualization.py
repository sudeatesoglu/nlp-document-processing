import spacy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from handle_document import read_document
from text_processing import preprocess_text

nlp = spacy.load("en_core_web_md")


def project_document_vectors(doc_text):
    doc = nlp(doc_text)
    sentences = [sent.text for sent in doc.sents]

    if len(sentences) < 2:
        print("The document should have at least two sentences.")
        return

    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    document_vectors = [nlp(sentence).vector for sentence in preprocessed_sentences]
    pca = PCA(n_components=2)
    transformed_vectors = pca.fit_transform(document_vectors)

    plt.figure(figsize=(10, 7))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1])

    sentences_dict = {}
    for i, sentence in enumerate(preprocessed_sentences):
        sentences_dict[f"Sentence {i + 1}"] = sentence
        plt.annotate(f"Sentence {i + 1}", xy=(transformed_vectors[i, 0], transformed_vectors[i, 1]))

    plt.title("PCA Projection of Document Vectors")
    plt.grid(True)
    return sentences_dict, plt


def pca_interface(doc_type, doc):
    doc_text = read_document(doc_type, doc)
    plt = project_document_vectors(doc_text)
    return plt
