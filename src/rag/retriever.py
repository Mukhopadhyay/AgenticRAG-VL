from langchain_qdrant import QdrantVectorStore
from typing import Optional
from config import settings
from langchain_core.documents import Document


class Retriever:

    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        search_type: Optional[str] = settings.SEARCH_TYPE,
        search_k: Optional[int] = settings.SEARCH_K,
    ):
        if not search_k:
            search_k = settings.SEARCH_K
        if not search_type:
            search_type = settings.SEARCH_TYPE

        self.vectorstore = vectorstore
        self.__retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": search_k,
            },
        )

    @property
    def retriever(self):
        return self.__retriever

    def retrieve(self, query: str) -> list[Document]:
        return self.__retriever.invoke(query)
