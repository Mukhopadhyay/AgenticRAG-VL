from pathlib import Path
from rich import print

from loaders import load_pdf
from rag.chunker import Chunker
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever

print("[bold magenta]AgenticRAG - ViralLens[/bold magenta]")

embedder = Embedder()
store = VectorStore(model=embedder.model)
retriver = Retriever(vectorstore=store.vectorstore)

path = Path("../data/raw")

for pdf_path in path.glob("*.pdf"):
    docs = load_pdf(pdf_path)

    chunker = Chunker()
    chunked_docs = chunker.process(docs)

    store.add_documents(chunked_docs)

query = "Describe the key finding about the petition from Facebook"

results = retriver.retrieve(query)
print(
    f"\n[bold cyan]Retrieved {len(results)} relevant chunks for query:[/bold cyan] [yellow]{query}[/yellow]"
)
for i, doc in enumerate(results, 1):
    print(f"\n[bold green]Result {i}:[/bold green]\n{doc.page_content}")
