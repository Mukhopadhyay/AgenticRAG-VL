from langchain_core.language_models import BaseChatModel
from config import settings
from rich import print

_PLAN_PROMPT = """
You are a query optimisation expert. Rewrite the user query to 
make it more specific and retrieval-friendly for a document collection

Rules:
- Return ONLY the rewritten query — nothing else.
- Keep it concise (one sentence).
- Do not answer the question.
{retry_hint}
Original query: {query}

Rewritten query:"""


class Planner:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def plan(self, query: str, retry_count: int = 0) -> str:
        retry_hint = (
            "\nNote: A previous attempt found no relevant documents. "
            "Try a different angle or rephrase."
            if retry_count > 0
            else ""
        )
        prompt = _PLAN_PROMPT.format(query=query, retry_hint=retry_hint)
        response = self.llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "".join(
                (
                    item.get("text", "")
                    if isinstance(item, dict) and item.get("type") == "text"
                    else (item if isinstance(item, str) else "")
                )
                for item in content
            )

        if settings.VERBOSE:
            print(
                f'[bold blue]Planner[/bold blue] - Rewritten query: "{content.strip()}"'
            )

        return content.strip()
