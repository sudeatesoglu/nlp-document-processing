import PyPDF2
import docx


def read_document(doc_type, file_path):
    if doc_type == "PDF":
        return read_pdf(file_path)
    elif doc_type == "TXT":
        return read_txt(file_path)
    elif doc_type == "DOCX":
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported document type.")


def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])
