from langchain_core.documents import Document
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever as RAGRetriever
from rich import print
from config import settings


class Retriever:
    """Thin wrapper that exposes a single retrieve() method using the RAG stack."""

    def __init__(self):
        embedder = Embedder()
        store = VectorStore(model=embedder.model)
        self._retriever = RAGRetriever(vectorstore=store.vectorstore)

    def retrieve(self, query: str) -> list[Document]:
        result = self._retriever.retrieve(query)
        if settings.VERBOSE:
            print(
                f"[bold blue]Retriever[/bold blue] - Retrieved {len(result)} documents"
            )
        return self._retriever.retrieve(query)
