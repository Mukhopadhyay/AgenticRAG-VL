from config import settings
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rich import print as rich_print


def print(*args, **kwargs):
    if kwargs.get("tab"):
        kwargs.pop("tab")
        rich_print("\t[bold red]Embedder[/bold red]", *args, **kwargs)
    else:
        rich_print("[bold red]Embedder[/bold red]", *args, **kwargs)


class Embedder:
    def __init__(self, model_name: Optional[str] = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self.__model = HuggingFaceEmbeddings(
            model_name=self.model_name, encode_kwargs={"normalize_embeddings": True}
        )
        if settings.VERBOSE:
            print("Initialized embedder with model:", self.model_name)

    @property
    def model(self) -> HuggingFaceEmbeddings:
        return self.__model

    def process(self, documents: list[Document]) -> list[list[float]]:
        result = self.__model.embed_documents([doc.page_content for doc in documents])
        shape = (len(result), len(result[0])) if result else (0, 0)
        if settings.VERBOSE:
            print(
                "Processed",
                len(documents),
                "documents and generated",
                shape[0],
                "embeddings with dimension",
                str(shape),
                tab=True,
            )
        return result
