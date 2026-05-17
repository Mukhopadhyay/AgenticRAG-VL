from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from llm.client import get_llm
from typing import AsyncGenerator
from rich import print as rich_print


def print(*args, **kwargs):
    rich_print("[red bold]RAG-Pipeline[/red bold]", *args, **kwargs)


class RAGPipeline:

    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore(model=self.embedder.model)
        self.retriever = Retriever(vectorstore=self.store.vectorstore)
        self.llm = get_llm()

    def run(self, query: str):
        results = self.retriever.retrieve(query)

        if not results:
            return "No relevant documents were found for your query."

        context = "\n\n".join([doc.page_content for doc in results])

        prompt = f"""
You are a helpful assistant that answers questions
based on the following retrieved information from a
collection of documents.

IMPORTANT RULES:
- Answer ONLY from the provided context.
- If the answer is not present in the context, say:
  'The answer was not found in the retrieved documents.'
- Do not mention missing context.
- Do not mention retrieval systems.
- Do not explain your reasoning process.

Context:
{context}

Question: {query}
"""

        response = self.llm.invoke(prompt)

        if isinstance(response.content, str):
            return response.content

        if isinstance(response.content, list):
            # For reasoning model
            text_parts = []

            for item in response.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))

                elif isinstance(item, str):
                    text_parts.append(item)

            return "\n".join(text_parts).strip()

        return str(response.content)

    async def astream(self, query: str) -> AsyncGenerator[str, None]:
        results = self.retriever.retrieve(query)
        print(f"Retrieved {len(results)} relevant documents for the query.")

        if not results:
            yield "No relevant documents were found for your query."
            return

        context = "\n\n".join([doc.page_content for doc in results])

        prompt = f"""
You are a helpful assistant that answers questions
based on the following retrieved information from a
collection of documents.

IMPORTANT RULES:
- Answer ONLY from the provided context.
- If the answer is not present in the context, say:
  'The answer was not found in the retrieved documents.'
- Do not mention missing context.
- Do not mention retrieval systems.
- Do not explain your reasoning process.

Context:
{context}

Question: {query}
"""

        async for chunk in self.llm.astream(prompt):
            if not chunk.content:
                continue

            if isinstance(chunk.content, str):
                yield chunk.content

            elif isinstance(chunk.content, list):
                for item in chunk.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            yield item.get("text", "")
                    elif isinstance(item, str):
                        yield item
