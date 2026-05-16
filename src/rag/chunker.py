from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings
from rich import print as rich_print


def print(*args, **kwargs) -> None:
    rich_print("\t[bold blue]Chunker[/bold blue]", *args, **kwargs)


class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, documents: list[Document]) -> list[Document]:
        if settings.VERBOSE:
            print(
                "Performing chunking, with chunk size:",
                self.chunk_size,
                "and chunk overlap:",
                self.chunk_overlap,
                "initial Document count:",
                len(documents),
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunked_docs = splitter.split_documents(documents=documents)

        if settings.VERBOSE:
            print("Chunking complete, final Document count:", len(chunked_docs))
        return chunked_docs
