from typing import Dict, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_milvus import Milvus
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class CSEKnowledgeBaseQuerier:
    """Class responsible for querying the knowledge base."""
    
    def __init__(self, milvus_uri: str, collection_name: str, hf_token: str):
        """
        Initialize the Knowledge Base Querier.
        
        Args:
            milvus_uri: URI for Milvus/Zilliz Cloud connection
            collection_name: Name of the collection to query
            hf_token: HuggingFace API token
        """
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize LLM
        self.llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.3-70B-Instruct",
            huggingfacehub_api_token=hf_token
        )
        
        # Connect to existing vector store
        self.vectorstore = Milvus(
            embedding_function=self.embedding_model,
            collection_name=collection_name,
            connection_args={"uri": milvus_uri}
        )

    def query(self, 
              question: str, 
              filters: Optional[Dict] = None, 
              top_k: int = 5,
              template: Optional[str] = None) -> Dict:
        """
        Query the knowledge base.
        
        Args:
            question: Question to ask
            filters: Optional filters (e.g., {"company_name": "CompanyA", "year": 2023})
            top_k: Number of relevant chunks to retrieve
            template: Optional custom prompt template
        """
        # Create search kwargs with filters if provided
        search_kwargs = {"k": top_k}
        if filters:
            search_kwargs["filter"] = filters

        # Create retriever with search parameters
        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        # Use default or custom prompt template
        if template is None:
            template = """You are a financial analyst assistant. Using only the context provided below, 
            answer the following question. If the information isn't in the context, say so.
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a detailed analysis with specific numbers and facts from the context when available.
            Answer:
            """
        
        PROMPT = PromptTemplate.from_template(template)
        
        # Create and execute chain
        document_chain = create_stuff_documents_chain(self.llm, PROMPT)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            "question": question,
        })
        
        return response

    def get_query_context(self, question: str, filters: Optional[Dict] = None, top_k: int = 5) -> List:
        """Get the context chunks that would be used to answer a question."""
        search_kwargs = {"k": top_k}
        if filters:
            search_kwargs["filter"] = filters
            
        docs = self.vectorstore.similarity_search(question, **search_kwargs)
        return docs
    