from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from rich import print as rich_print
from config import settings


def print(*args, **kwargs) -> None:
    rich_print("\t[bold green]PDF-Loader[/bold green]", *args, **kwargs)


def load_pdf(file_path: str) -> list[Document]:
    if settings.VERBOSE:
        print("Loading PDF from path:", file_path)
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()
    return docs
