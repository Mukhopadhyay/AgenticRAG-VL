from pathlib import Path
from rich import print
from config import settings

from loaders import load_pdf
from rag.chunker import Chunker

print("[bold magenta]AgenticRAG - ViralLens[/bold magenta]")

path = Path("../data/raw")
for pdf_path in path.glob("*.pdf"):
    docs = load_pdf(pdf_path)
    # if settings.VERBOSE:
    #     print(docs)

    chunker = Chunker()
    chunked_docs = chunker.process(docs)
