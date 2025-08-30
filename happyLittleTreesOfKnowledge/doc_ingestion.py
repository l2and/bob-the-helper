import os
import logging
from typing import List
from langchain_community.document_loaders import GitHubLoader
from langchain_community.vectorstores import FAISS
from langchain_anthropic import AnthropicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentationIngestion:
    """Handle ingestion of LangChain documentation"""
    
    def __init__(self):
        self.embeddings = AnthropicEmbeddings(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_langchain_docs(self) -> List[Document]:
        """Load LangChain documentation from GitHub"""
        try:
            logger.info("Loading LangChain documentation from GitHub...")
            
            # Load core LangChain documentation
            loader = GitHubLoader(
                clone_url="https://github.com/langchain-ai/langchain",
                repo_path="docs/docs",
                branch="master",
                file_filter=lambda file_path: file_path.endswith(".md") or file_path.endswith(".mdx")
            )
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from LangChain repo")
            
            # Also load LangGraph documentation
            try:
                langgraph_loader = GitHubLoader(
                    clone_url="https://github.com/langchain-ai/langgraph",
                    repo_path="docs",
                    branch="main",
                    file_filter=lambda file_path: file_path.endswith(".md") or file_path.endswith(".mdx")
                )
                
                langgraph_docs = langgraph_loader.load()
                documents.extend(langgraph_docs)
                logger.info(f"Added {len(langgraph_docs)} LangGraph documents")
                
            except Exception as e:
                logger.warning(f"Could not load LangGraph docs: {e}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documentation: {e}")
            # Return fallback documentation
            return self.create_fallback_docs()
    
    def create_fallback_docs(self) -> List[Document]:
        """Create fallback documentation when GitHub loading fails"""
        logger.info("Creating fallback documentation...")
        
        fallback_content = {
            "LangChain Basics": """
            LangChain is a framework for developing applications powered by language models. 
            It provides tools for connecting LLMs to other sources of data and allowing them 
            to interact with their environment. Key components include:
            - Prompts: Templates for formatting input to language models
            - Models: Various language model integrations
            - Chains: Sequences of calls to models and other utilities
            - Agents: Systems that use language models to choose actions
            """,
            
            "LangGraph Introduction": """
            LangGraph is a library for building stateful, multi-step applications with LLMs. 
            It extends LangChain Expression Language with the ability to coordinate multiple chains 
            across multiple steps of computation. Key features:
            - State management across multiple steps
            - Conditional branching and looping
            - Built-in persistence and memory
            - Integration with LangSmith for monitoring
            """,
            
            "LCEL (LangChain Expression Language)": """
            LCEL is a declarative way to compose chains in LangChain. It provides:
            - Pipe operator (|) for chaining components
            - Automatic input/output schema validation
            - Streaming support
            - Async support
            - Optimized parallel execution
            Example: chain = prompt | model | output_parser
            """,
            
            "Vector Stores": """
            Vector stores are databases optimized for storing and querying high-dimensional vectors.
            In LangChain, they're commonly used for:
            - Semantic search over documents
            - Retrieval-Augmented Generation (RAG)
            - Finding similar content
            Popular implementations: FAISS, Pinecone, Chroma, Weaviate
            """,
            
            "Common Error Solutions": """
            ImportError fixes:
            - pip install langchain langchain-anthropic
            - Check Python version compatibility
            - Use virtual environments
            
            API Key issues:
            - Set environment variables correctly
            - Check API key validity
            - Verify rate limits
            
            Memory issues:
            - Use text splitters for large documents
            - Implement chunking strategies
            - Consider batch processing
            """
        }
        
        documents = []
        for title, content in fallback_content.items():
            doc = Document(
                page_content=content,
                metadata={"source": f"fallback_{title.lower().replace(' ', '_')}", "title": title}
            )
            documents.append(doc)
        
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a FAISS vector store from documents"""
        try:
            logger.info("Splitting documents into chunks...")
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(splits)} document chunks")
            
            logger.info("Creating vector embeddings...")
            vector_store = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            logger.info("Vector store created successfully!")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vector_store(self, vector_store: FAISS, path: str = "vector_store"):
        """Save vector store to disk"""
        try:
            vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load_vector_store(self, path: str = "vector_store") -> FAISS:
        """Load vector store from disk"""
        try:
            vector_store = FAISS.load_local(path, self.embeddings)
            logger.info(f"Vector store loaded from {path}")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

def main():
    """Main function to ingest documentation"""
    ingestion = DocumentationIngestion()
    
    # Load documents
    documents = ingestion.load_langchain_docs()
    
    # Create vector store
    vector_store = ingestion.create_vector_store(documents)
    
    # Save for later use
    ingestion.save_vector_store(vector_store)
    
    # Test the vector store
    test_query = "What is LangGraph?"
    results = vector_store.similarity_search(test_query, k=3)
    
    print(f"\nTest query: {test_query}")
    print(f"Found {len(results)} relevant documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('title', 'Unknown')}")
        print(doc.page_content[:200] + "...")

if __name__ == "__main__":
    main()