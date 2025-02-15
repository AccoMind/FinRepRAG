from typing import Dict, Optional
from langchain_milvus import Milvus
from langchain.embeddings.base import Embeddings

class MilvusManager:
    """Manages connections and operations with Milvus/Zilliz vector store."""
    
    def __init__(self,
                 milvus_uri: str,
                 milvus_user: str,
                 milvus_password: str,
                 embedding_model: Embeddings,
                 collection_name: str):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.connection_args = {
            "uri": milvus_uri,
            "user": milvus_user,
            "password": milvus_password,
            "secure": True
        }
        self.vectorstore = self._connect_vectorstore()

    def _connect_vectorstore(self) -> Optional[Milvus]:
        """Establish connection to vector store."""
        try:
            return Milvus(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                auto_id=True
            )
        except Exception as e:
            print(f"Error connecting to Milvus: {str(e)}")
            return None

    def create_vectorstore(self, documents: list) -> Milvus:
        """Create a new vector store with documents."""
        return Milvus.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=self.collection_name,
            connection_args=self.connection_args,
            index_params={"index_type": "FLAT"}
        )

    def add_documents(self, documents: list) -> None:
        """Add documents to existing vector store."""
        if self.vectorstore:
            self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, filters: Optional[Dict] = None, k: int = 5):
        """Perform similarity search."""
        search_kwargs = {"k": k}
        if filters:
            search_kwargs["filter"] = filters
        return self.vectorstore.similarity_search(query, **search_kwargs)

    def get_retriever(self, search_kwargs: Dict):
        """Get retriever with specified search parameters."""
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        if not self.vectorstore:
            return {"status": "Not connected"}
        try:
            stats = self.vectorstore.col.get_statistics()
            return {
                "collection_name": self.collection_name,
                "total_documents": stats.get("row_count", "Unknown"),
                "embedding_dimension": stats.get("dim", "Unknown")
            }
        except Exception as e:
            return {"error": f"Failed to retrieve stats: {str(e)}"}