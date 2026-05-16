from pathlib import Path
from rich import print
from loaders import load_pdf

path = Path("../data/raw")
for pdf_path in path.glob("*.pdf"):
    print("[bold green]Loading PDF:[/bold green]", pdf_path)
    docs = load_pdf(pdf_path)
    print(docs)
