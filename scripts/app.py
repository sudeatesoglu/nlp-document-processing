import gradio as gr
from similarity import *
from entity_recognition import ner_interface
from pca_visualization import pca_interface

similarity_demo = gr.Interface(
    fn=check_doc_similarity,
    inputs=[
        gr.Radio(["PDF", "TXT", "DOCX"], label="Document Type"),
        gr.File(label="Document 1"),
        gr.File(label="Document 2"),
    ],
    outputs=gr.Textbox(label="Similarity Score"),
    title="Document Similarity Detector",
    description="Upload two documents to check their similarity.",
)

categorical_similarity_demo = gr.Interface(
    fn=check_categorical_similarity,
    inputs=[
        gr.Textbox(label="Category"),
        gr.Radio(["PDF", "TXT", "DOCX"], label="Document Type"),
        gr.File(label="Document")
    ],
    outputs=gr.Textbox(label="Categorical Similarity Scores"),
    title="Categorical Similarity Checker",
    description="Check the similarity of documents to a given category."
)

ner_demo = gr.Interface(
    fn=ner_interface,
    inputs=[
        gr.Radio(["PDF", "TXT", "DOCX"], label="Document Type"),
        gr.File(label="Document")
    ],
    outputs=[
        gr.Textbox(label="Named Entities"),
    ],
    title="Named Entity Recognition (NER)",
    description="Extract and display named entities from documents."
)

pca_demo = gr.Interface(
    fn=pca_interface,
    inputs=[
        gr.Radio(["PDF", "TXT", "DOCX"], label="Document Type"),
        gr.File(label="Document")
    ],
    outputs=gr.Plot(),
    title="Document PCA Visualization",
    description="Visualize document vectors using PCA."
)

demo = gr.TabbedInterface(
    interface_list=[
        similarity_demo,
        categorical_similarity_demo,
        ner_demo,
        pca_demo
    ],
    tab_names=[
        "Document Similarity",
        "Categorical Similarity",
        "Named Entity Recognition",
        "PCA Visualization"
    ]
)


if __name__ == "__main__":
    demo.launch()
