from qdrant_client import QdrantClient
from config import settings
import qdrant_client
from qdrant_client.http import models as rest
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from rich import print as rich_print


def print(*args, **kwargs):
    rich_print("\t[bold blue]VectorStore[/bold blue]", *args, **kwargs)


class VectorStore:

    def __init__(
        self,
        model: HuggingFaceEmbeddings,
        collection_name: Optional[str] = settings.QDRANT_COLLECTION,
    ):
        self.client = QdrantClient(url=settings.QDRANT_URL)

        self.model = model
        self.collection_name = collection_name
        # self.recreate_collection()

        if self.is_collection_exists():
            print(f"Collection '{self.collection_name}' already exists.")
        else:
            print(
                f"Collection '{self.collection_name}' does not exist. Creating a new collection."
            )
            self.recreate_collection()

        self.__vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.model,
        )

        # try:
        #     self.__vectorstore = QdrantVectorStore(
        #         client=self.client,
        #         collection_name=self.collection_name,
        #         embedding=self.model,
        #     )
        # except Exception:
        #     # If the collection doesn't exist, create it and then initialize the vectorstore
        #     self.recreate_collection()
        #     self.__vectorstore = QdrantVectorStore(
        #         client=self.client,
        #         collection_name=self.collection_name,
        #         embedding=self.model,
        #     )

    @property
    def vectorstore(self):
        return self.__vectorstore

    def is_collection_exists(self) -> bool:
        try:
            self.client.get_collection(collection_name=self.collection_name)
            return True
        except Exception:
            return False

    def collection_data_count(self) -> int:
        try:
            x = self.client.count(collection_name=self.collection_name, exact=True)
            return x.count
        except Exception as err:
            print("Error:", err)
            return 0

    def recreate_collection(self):
        __embedding_size = len(self.model.embed_query("test"))

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={"size": __embedding_size, "distance": rest.Distance.COSINE},
        )
        if settings.VERBOSE:
            print(
                f'[green]Recreated collection "{self.collection_name}" with embedding size {__embedding_size}.[/green]'
            )

    def add_documents(self, documents: list[str]):
        self.__vectorstore.add_documents(documents)
        if settings.VERBOSE:
            print(
                f"[green]Added {len(documents)} documents to the vector store.[/green]"
            )
