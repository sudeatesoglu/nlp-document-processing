# Document Processing Tool

Document Processing tool is a simple NLP project that provides comparison of two documents in different formats (pdf, txt, or docx) in terms of similarity score calculation, categorical similarity detection,
highlighting similar words between PDF documents, named entity recognition (NER), and PCA visualization.

## Features

- **Similarity Score Calculation**: Calculates the similarity score between two documents.
- **Categorical Similarity**: Detects the similarity score of a document based on given keyword.
- **Highlighting Similar Words**: Highlights similar words between two PDF documents.
- **Named Entity Recognition**: Extracts and displays named entities from documents.
- **PCA Visualization**: Visualizes document word vectors using PCA.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sudeatesoglu/nlp-document-processing.git
    cd nlp-document-processing
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Launch the application:
    ```bash
    python app.py
    ```
