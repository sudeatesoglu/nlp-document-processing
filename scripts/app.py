import gradio as gr
from similarity import *
from entity_recognition import ner_interface
from pca_visualization import pca_interface

similarity_demo = gr.Interface(
    fn=check_cosine_similarity,
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

highlight_pattern_demo = gr.Interface(
    fn=pattern_interface,
    inputs=[
        gr.Textbox(label="Word to Search"),
        gr.Radio(["PDF"], label="Document Type"),
        gr.File(label="PDF Document")
    ],
    outputs=[
        gr.File(label="Highlighted PDF"),
        gr.JSON(label="Matched Span")
    ],
    title="Pattern Highlighter",
    description="Upload a PDF document to search and highlight the given pattern."
)

highlight_similar_words_demo = gr.Interface(
    fn=highlight_similar_words,
    inputs=[
        gr.File(label="PDF 1"),
        gr.File(label="PDF 2")
    ],
    outputs=[
        gr.File(label="Highlighted PDF 1"),
        gr.File(label="Highlighted PDF 2")
    ],
    title="Highlight Similar Words in PDFs",
    description="Upload two PDF documents to highlight similar words between them."
)

ner_demo = gr.Interface(
    fn=ner_interface,
    inputs=[
        gr.Radio(["PDF", "TXT", "DOCX"], label="Document Type"),
        gr.File(label="Document")
    ],
    outputs=[
        gr.Textbox(label="Named Entities"),
        gr.Textbox(label="Entity Highlights"),
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
        highlight_pattern_demo,
        highlight_similar_words_demo,
        ner_demo,
        pca_demo
    ],
    tab_names=[
        "Document Similarity",
        "Categorical Similarity",
        "Highlight Pattern",
        "Highlight Similar Words",
        "Named Entity Recognition",
        "PCA Visualization"
    ]
)

if __name__ == "__main__":
    demo.launch()
