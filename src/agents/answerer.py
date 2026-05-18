from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from config import settings
from rich import print

_ANSWER_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.

Rules:
- Be concise and direct.
- If the answer is not in the context, say exactly:
  'The answer was not found in the retrieved documents.'
- Do not mention retrieval systems or missing context.

Context:
{context}

Question: {query}

Answer:"""


class Answerer:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def answer(self, query: str, docs: list[Document]) -> str:
        if not docs:
            return "The answer was not found in the retrieved documents."

        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = _ANSWER_PROMPT.format(context=context, query=query)
        response = self.llm.invoke(prompt)

        content = response.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            ).strip()

        if settings.VERBOSE:
            print(f"[bold blue]Answerer[/bold blue] - Generated answer: {content}")

        return str(content)
