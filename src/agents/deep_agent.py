from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool as lc_tool
from langchain_core.language_models import BaseChatModel
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever as RAGRetriever
from agents.grader import Grader
from rich import print
from config import settings
from llm.client import get_raw_llm

_SYSTEM_PROMPT = """
You are an expert research assistant specializing in
legal documents and SEC filings with access to a curated document collection.

Your job is to answer the user's question as accurately and completely as possible.

## `search_documents`

Use this tool to search the document collection for information relevant to the question.
- You may call it multiple times with different phrasings to improve coverage.
- Try broad queries first, then narrow down with more specific follow-ups.
- Combine evidence from multiple search results to form a coherent answer.

## Guidelines

- Base your answer strictly on what the documents say.
- If the documents contain only partial information, share what you found and note the gap.
- Do not hallucinate facts not present in the retrieved excerpts.
"""


def _build_search_tool(rag_retriever: RAGRetriever, grader: Grader):
    """Return a LangChain tool wrapping the RAG retriever + grader."""

    @lc_tool
    def search_documents(query: str) -> str:
        """Search the document collection for information relevant to a query.

        Retrieves semantically similar document chunks from the vector store and
        filters them for relevance, returning the matching excerpts as text.

        Args:
            query: Natural-language search query.

        Returns:
            Concatenated relevant document excerpts, or a message when nothing
            relevant is found.
        """
        docs = rag_retriever.retrieve(query)
        relevant = grader.grade(query=query, docs=docs)
        if not relevant:
            return "No relevant documents found for this query."
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in relevant
        )

    return search_documents


class DeepAgent:
    """A LangGraph ReAct agent that uses the project's RAG stack as tools.

    It is intended as an escalation path: when the structured LangGraph retry loop
    is insufficient, the DeepAgent can plan and run multiple targeted searches before
    synthesising a final answer.
    """

    def __init__(self, llm: BaseChatModel):
        embedder = Embedder()
        store = VectorStore(model=embedder.model)
        rag_retriever = RAGRetriever(vectorstore=store.vectorstore)
        grader = Grader(llm=llm)

        search_tool = _build_search_tool(rag_retriever, grader)

        self._agent = create_react_agent(
            model=get_raw_llm(),
            tools=[search_tool],
            prompt=_SYSTEM_PROMPT,
        )

    def run(self, query: str) -> str:
        """Invoke the deep agent and return the final answer string."""
        result = self._agent.invoke({"messages": [{"role": "user", "content": query}]})
        last_message = result["messages"][-1]
        content = last_message.content

        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )

        if settings.VERBOSE:
            print(
                f"[bold cyan]DeepAgent[/bold cyan] - Produced answer ({len(content)} chars)"
            )

        return content.strip()
