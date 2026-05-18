from typing import TypedDict, List
from langchain_core.documents import Document


class State(TypedDict):
    query: str
    rewritten_query: str
    retrieved_docs: List[Document]
    filtered_docs: List[Document]
    answer: str
    verified: bool
    retry_count: int
    deep_agent_used: bool
