from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()
    return docs
