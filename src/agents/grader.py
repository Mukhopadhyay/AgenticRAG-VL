from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from config import settings
from rich import print

_GRADE_PROMPT = """You are a relevance grader.

Query: {query}

Document excerpt:
{excerpt}

Is this document relevant to the query? Answer with ONLY 'yes' or 'no':"""


class Grader:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def grade(self, query: str, docs: list[Document]) -> list[Document]:
        relevant: list[Document] = []
        for doc in docs:
            prompt = _GRADE_PROMPT.format(
                query=query,
                excerpt=doc.page_content[:1000],
            )
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
            if "yes" in content.strip().lower():
                relevant.append(doc)

        if settings.VERBOSE:
            print(
                f"[bold green]Grader[/bold green] - Found {len(relevant)} relevant documents out of {len(docs)}."
            )
        return relevant
