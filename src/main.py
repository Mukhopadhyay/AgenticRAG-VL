# from pathlib import Path
from rich import print

from loaders import load_pdf
from rag.chunker import Chunker
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from llm.client import get_llm

from scripts.ingest import ingest

print("[bold magenta]AgenticRAG - ViralLens[/bold magenta]")


# path = Path("../data/raw")

# for pdf_path in path.glob("*.pdf"):
#     docs = load_pdf(pdf_path)

#     chunker = Chunker()
#     chunked_docs = chunker.process(docs)

#     store.add_documents(chunked_docs)

ingest()

# embedder = Embedder()
# store = VectorStore(model=embedder.model)
# retriver = Retriever(vectorstore=store.vectorstore)


# query = "Describe the key finding about the petition from Facebook"

# results = retriver.retrieve(query)
# print(
#     f"\n[bold cyan]Retrieved {len(results)} relevant chunks for query:[/bold cyan] [yellow]{query}[/yellow]"
# )
# for i, doc in enumerate(results, 1):
#     print(f"\n[bold green]Result {i}:[/bold green]\n{doc.page_content}")

# context = "\n\n".join([doc.page_content for doc in results])

# prompt = f"""
# You are a helpful assistant that answers questions
# based on the following retrieved information from a
# collection of documents.

# **NOTE**: Use only the retrieved information to answer the question, and do not make up any information.

# {context}
# """

# llm = get_llm()
# response = llm.invoke(prompt + "\n\nQuestion: " + query)

# print(f"\n[bold magenta]LLM Response:[/bold magenta]\n{response}")
