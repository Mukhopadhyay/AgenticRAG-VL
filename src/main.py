from pathlib import Path
from rich import print

from loaders import load_pdf
from rag.chunker import Chunker
from rag.embedder import Embedder
from rag.vectorstore import VectorStore

print("[bold magenta]AgenticRAG - ViralLens[/bold magenta]")

embedder = Embedder()
store = VectorStore(model=embedder.model)


path = Path("../data/raw")

for pdf_path in path.glob("*.pdf"):
    docs = load_pdf(pdf_path)

    chunker = Chunker()
    chunked_docs = chunker.process(docs)

    store.add_documents(chunked_docs)
