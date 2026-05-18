from rich import print

from scripts.ingest import ingest
from graph.workflow import workflow

print("[bold magenta]AgenticRAG - ViralLens[/bold magenta]")

ingest()

query = "Tell me about the facebook petition"

print(
    f"\n[bold cyan]Running agentic RAG for query:[/bold cyan] [yellow]{query}[/yellow]"
)

result = workflow.invoke({"query": query, "retry_count": 0})

print(f"\n[bold green]Rewritten query:[/bold green] {result.get('rewritten_query')}")
print(f"[bold blue]Retrieved docs:[/bold blue] {len(result.get('retrieved_docs', []))}")
print(f"[bold blue]Filtered docs:[/bold blue] {len(result.get('filtered_docs', []))}")
print(f"[bold yellow]Verified:[/bold yellow] {result.get('verified')}")
print(f"\n[bold magenta]Answer:[/bold magenta]\n{result.get('answer')}")
