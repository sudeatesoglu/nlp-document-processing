import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load("en_core_web_md")


def project_word_vectors(words):
    word_ids = [nlp.vocab.strings[word] for word in words]
    word_vectors = np.vstack([nlp.vocab.vectors[i] for i in word_ids])
    pca = PCA(n_components=2)
    transformed_vectors = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 7))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(transformed_vectors[i, 0], transformed_vectors[i, 1]))
    plt.title("Word Vectors PCA Projection")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()
