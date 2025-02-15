from typing import Dict, List, Optional
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_milvus import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

class KnowledgeBaseQuerier:
    """Class responsible for querying a knowledge base of Annual Reports from companies listed in the Colombo Stock Exchange (CSE)."""

    def __init__(self,
                 milvus_uri: str,
                 milvus_user: str,
                 milvus_password: str,
                 collection_name: str,
                 hf_token: str):
        """
        Initialize the Knowledge Base Querier.

        Args:
            milvus_uri: URI for Milvus/Zilliz Cloud connection
            milvus_user: Milvus username
            milvus_password: Milvus password
            collection_name: Name of the collection to query
            hf_token: HuggingFace API token
        """
        self.collection_name = collection_name

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize LLM
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=hf_token,
            task="text-generation"  # Ensure task is explicitly set
        )

        # Setup Milvus connection
        self.connection_args = {
            "uri": milvus_uri,
            "user": milvus_user,
            "password": milvus_password,
            "secure": True
        }

        # Connect to existing vector store
        try:
            self.vectorstore = Milvus(
                embedding_function=self.embedding_model,
                collection_name=collection_name,
                connection_args=self.connection_args
            )
            print(f"Successfully connected to collection: {collection_name}")
        except Exception as e:
            raise Exception(f"Failed to connect to Milvus: {str(e)}")

    def query(self,
              question: str,
              filters: Optional[Dict] = None,
              top_k: int = 5,
              template: Optional[str] = None) -> Dict:
        """
        Query the CSE knowledge base for Annual Reports.

        Args:
            question: Question to ask
            filters: Optional filters (e.g., {"company_name": "CompanyA", "year": 2023})
            top_k: Number of relevant chunks to retrieve
            template: Optional custom prompt template
        """
        search_kwargs = {"k": top_k}
        if filters:
            search_kwargs["filter"] = filters

        # Create retriever with search parameters
        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        # Default or custom prompt template
        if template is None:
            template = """You are a financial analyst specializing in corporate annual reports. Using only the context provided below,
            answer the following question accurately. If the information is not in the context, explicitly state that.

            Context:
            {context}

            Question: {input}

            Provide a detailed response with specific numbers and insights from the report.
            Answer:
            """

        PROMPT = PromptTemplate.from_template(template)

        # Create and execute chain
        document_chain = create_stuff_documents_chain(self.llm, PROMPT)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({
            "input": question,
        })

        return response

    def get_query_context(self, question: str, filters: Optional[Dict] = None, top_k: int = 5) -> List:
        """
        Retrieve the most relevant context chunks that would be used to answer a question.
        """
        search_kwargs = {"k": top_k}
        if filters:
            search_kwargs["filter"] = filters

        docs = self.vectorstore.similarity_search(question, **search_kwargs)
        return docs

    def get_collection_info(self) -> Dict:
        """Get metadata about the connected collection."""
        try:
            collection = self.vectorstore.col
            stats = collection.get_statistics()
            return {
                "collection_name": self.collection_name,
                "total_documents": stats.get("row_count", "Unknown"),
                "embedding_dimension": stats.get("dim", "Unknown")
            }
        except Exception as e:
            return {"error": f"Failed to retrieve collection info: {str(e)}"}
