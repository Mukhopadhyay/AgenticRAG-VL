from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from config import settings
from rich import print

_VERIFY_PROMPT = """
You are a fact-checking assistant.
Decide whether the answer is fully supported 
by the context dcuments.

Query: {query}

Context (truncated):
{context}

Answer: {answer}

Is the answer fully supported by the context? Answer with ONLY 'yes' or 'no':"""


class Verifier:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def verify(self, query: str, docs: list[Document], answer: str) -> bool:
        if not docs:
            return False
        context = "\n\n".join(doc.page_content for doc in docs)[:3000]
        prompt = _VERIFY_PROMPT.format(query=query, context=context, answer=answer)
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
                f"[bold magenta]Verifier[/bold magenta] - Verification response: {content}"
            )

        return "yes" in content.strip().lower()
