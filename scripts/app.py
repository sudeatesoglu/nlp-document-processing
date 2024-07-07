import gradio as gr
from similarity import check_doc_similarity
from visualization import project_word_vectors

demo1 = gr.Interface(
    fn=check_doc_similarity,
    inputs=[
        gr.Radio(["PDF", "TXT"], label="Document Type"),
        gr.File(label="Document 1"),
        gr.File(label="Document 2"),
    ],
    outputs=gr.Textbox(label="Similarity Score"),
    title="Document Similarity Checker",
    description="Upload two documents to compare their similarity.",
)

demo2 = gr.Interface(
    fn=project_word_vectors,
    inputs=gr.Textbox(label="Words to Visualize"),
    outputs=None,
    title="Word Vector Visualization",
    description="Enter words separated by space to visualize their word vectors.",
)

demo = gr.TabbedInterface([demo1, demo2], ["Document Comparison", "Word Vector Visualization"])

if __name__ == "__main__":
    demo.launch()
