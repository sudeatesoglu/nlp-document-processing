import spacy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from handle_document import read_document
from text_processing import preprocess_text

nlp = spacy.load("en_core_web_md")


def project_document_vectors(doc_text):
    sentences = [sent.text for sent in nlp(doc_text).sents]
    document_vectors = [nlp(sentence).vector for sentence in sentences]
    pca = PCA(n_components=2)
    transformed_vectors = pca.fit_transform(document_vectors)

    plt.figure(figsize=(10, 7))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1])
    for i, sentence in enumerate(sentences):
        plt.annotate(f"Sentence {i + 1}", xy=(transformed_vectors[i, 0], transformed_vectors[i, 1]))
    plt.title("Document Vectors PCA")
    plt.grid(True)
    plt.show()
    plt.close()
    return plt


def pca_interface(doc_type, doc):
    doc_text = read_document(doc_type, doc.name)
    doc_text = preprocess_text(doc_text)
    plt = project_document_vectors(doc_text)
    return plt
