from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.chunker import Chunker
from langchain_core.documents import Document

from loaders import load_pdf

from pathlib import Path
from typing import Optional


def __read_documents(doc_path: Optional[str] = "../data/raw") -> list[Document]:
    path = Path(doc_path)
    documents = []
    for pdf_path in path.glob("*.pdf"):
        docs = load_pdf(pdf_path)
        documents.extend(docs)
    return documents


def ingest():
    documents = __read_documents()

    # Chunk documents
    chunker = Chunker()
    chunked_docs = chunker.process(documents)

    # Store in vector database
    embedder = Embedder()
    store = VectorStore(model=embedder.model)

    # Check if the collection exists
    if not store.is_collection_exists():
        print(
            f"Collection '{store.collection_name}' does not exist. Creating a new collection."
        )
        store.recreate_collection()
        store.add_documents(chunked_docs)

    else:
        print(f"Collection '{store.collection_name}' already exists.")
        existing_count = store.collection_data_count()
        print(
            f"Collection '{store.collection_name}' already has {existing_count} documents."
        )

        if existing_count != len(chunked_docs):
            print(
                f"Existing document count ({existing_count}) does not match the new document count ({len(chunked_docs)}). Recreating collection."
            )
            store.recreate_collection()
            store.add_documents(chunked_docs)
