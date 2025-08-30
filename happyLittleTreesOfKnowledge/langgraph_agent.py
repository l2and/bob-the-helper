import os
import json
from typing import TypedDict, Literal, Annotated, List, Optional
from langchain_anthropic import ChatAnthropic, AnthropicEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import GitHubLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langsmith import Client
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentationAssistantState(TypedDict):
    """State for our Bob Ross Documentation Assistant"""
    original_text: str
    query_type: Optional[Literal["code_explanation", "error_help", "concept_learning", "api_usage", "general_help"]]
    context_documents: List[str]
    bob_ross_response: str
    confidence_score: float
    citations: List[str]
    processing_steps: List[str]

class BobRossDocumentationAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-20250514",
            temperature=0.1
        )
        
        self.embeddings = AnthropicEmbeddings(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Initialize LangSmith client
        self.langsmith_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        
        # Load the Bob Ross prompt from LangSmith 
        try:
            self.bob_ross_prompt = self.langsmith_client.pull_prompt("l2and/bob_ross_help")
            logger.info("Successfully loaded Bob Ross prompt from LangSmith")
        except Exception as e:
            logger.warning(f"Could not load LangSmith prompt: {e}")
            self.bob_ross_prompt = None
        
        self.vector_store = None
        self.setup_documentation_store()
        self.graph = self.create_langgraph_workflow()
    
    def setup_documentation_store(self):
        """Set up the documentation vector store with LangChain docs"""
        try:
            # For demo purposes, we'll use a subset of LangChain docs
            # In production, you'd want to load the full documentation
            logger.info("Setting up documentation vector store...")
            
            # Sample documents - in real implementation, load from GitHub
            sample_docs = [
                "LangGraph is a library for building stateful, multi-step applications with LLMs. It provides a way to create agents and workflows that can maintain state, loop, and make conditional decisions.",
                "LCEL (LangChain Expression Language) is a declarative way to compose chains. You can use the pipe operator | to chain components together.",
                "LangSmith is a platform for building production-ready LLM applications. It provides tools for debugging, testing, and monitoring LLM applications.",
                "Vector stores in LangChain allow you to store and retrieve documents based on semantic similarity. Common implementations include FAISS, Pinecone, and Chroma.",
                "Document loaders in LangChain help you load documents from various sources like PDFs, web pages, databases, and APIs.",
                "Text splitters help break down large documents into smaller chunks that fit within LLM context windows while maintaining semantic coherence."
            ]
            
            # Create vector store
            self.vector_store = FAISS.from_texts(
                texts=sample_docs,
                embedding=self.embeddings
            )
            logger.info("Documentation vector store created successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup documentation store: {e}")
            self.vector_store = None
    
    def classify_query(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Classify the type of help request"""
        try:
            text = state["original_text"]
            steps = state.get("processing_steps", [])
            steps.append("🎨 Analyzing your text to understand what kind of help you need...")
            
            # Use Claude to classify the query
            classification_prompt = f"""
            Analyze this highlighted text and classify what type of help the user likely needs:
            
            Text: "{text}"
            
            Choose ONE of these categories:
            - code_explanation: User wants to understand what code does
            - error_help: User has an error message they need help with
            - concept_learning: User wants to learn about a concept or term
            - api_usage: User wants to know how to use an API or function
            - general_help: General questions or unclear requests
            
            Respond with just the category name.
            """
            
            response = self.llm.invoke([HumanMessage(content=classification_prompt)])
            query_type = response.content.strip().lower()
            
            if query_type not in ["code_explanation", "error_help", "concept_learning", "api_usage", "general_help"]:
                query_type = "general_help"
            
            logger.info(f"Classified query as: {query_type}")
            
            return {
                **state,
                "query_type": query_type,
                "processing_steps": steps
            }
            
        except Exception as e:
            logger.error(f"Error in classify_query: {e}")
            return {
                **state,
                "query_type": "general_help",
                "processing_steps": steps + ["Had trouble classifying your request, but I'll give it my best shot!"]
            }
    
    def retrieve_documentation(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Retrieve relevant documentation based on the query"""
        try:
            text = state["original_text"]
            query_type = state["query_type"]
            steps = state.get("processing_steps", [])
            steps.append("🔍 Searching through documentation to find relevant information...")
            
            if not self.vector_store:
                logger.warning("No vector store available")
                return {
                    **state,
                    "context_documents": [],
                    "processing_steps": steps
                }
            
            # Create a search query based on the original text and query type
            search_query = f"{text} {query_type}"
            
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(
                search_query, 
                k=3
            )
            
            context_documents = [doc.page_content for doc in docs]
            citations = [f"LangChain Documentation - Section {i+1}" for i in range(len(docs))]
            
            logger.info(f"Retrieved {len(context_documents)} relevant documents")
            
            return {
                **state,
                "context_documents": context_documents,
                "citations": citations,
                "processing_steps": steps
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_documentation: {e}")
            return {
                **state,
                "context_documents": [],
                "citations": [],
                "processing_steps": steps
            }
    
    def generate_bob_ross_response(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Generate the final Bob Ross-style response"""
        try:
            text = state["original_text"]
            query_type = state["query_type"]
            context_docs = state.get("context_documents", [])
            steps = state.get("processing_steps", [])
            steps.append("🎨 Painting you a beautiful explanation with Bob Ross wisdom...")
            
            # Prepare context
            context = "\n".join(context_docs) if context_docs else "No specific documentation found."
            
            # Create the prompt for Bob Ross response
            if self.bob_ross_prompt:
                # Use the LangSmith prompt
                response = (self.bob_ross_prompt | self.llm).invoke({
                    "text": text,
                    "context": context,
                    "query_type": query_type
                })
                bob_response = response.content
            else:
                # Fallback prompt
                fallback_prompt = f"""
                You are Bob Ross, the gentle painting instructor, now helping people understand LangChain concepts.
                
                User's highlighted text: "{text}"
                Query type: {query_type}
                Relevant documentation: {context}
                
                Respond in Bob Ross's encouraging, gentle style. Use painting metaphors to explain technical concepts.
                Make complex things feel approachable and reassuring. End with encouragement and next steps.
                """
                
                response = self.llm.invoke([HumanMessage(content=fallback_prompt)])
                bob_response = response.content
            
            # Calculate confidence score based on context availability
            confidence = 0.9 if context_docs else 0.6
            
            logger.info("Generated Bob Ross response successfully")
            
            return {
                **state,
                "bob_ross_response": bob_response,
                "confidence_score": confidence,
                "processing_steps": steps + ["✨ Your happy little explanation is ready!"]
            }
            
        except Exception as e:
            logger.error(f"Error in generate_bob_ross_response: {e}")
            fallback_response = """
            Well hello there, friend! I see you've highlighted some text, and that's just wonderful. 
            Even though I'm having a little technical difficulty right now, remember that every 
            mistake is just a happy little accident. Let's keep learning together, and don't forget - 
            you have the power to create beautiful code, just like a beautiful painting!
            """
            
            return {
                **state,
                "bob_ross_response": fallback_response,
                "confidence_score": 0.3,
                "processing_steps": steps + ["Had some technical difficulties, but made the best of it!"]
            }
    
    def should_continue_processing(self, state: DocumentationAssistantState) -> Literal["continue", "end"]:
        """Decide whether to continue processing or end"""
        # For this simple workflow, we always continue to completion
        return "continue"
    
    def create_langgraph_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(DocumentationAssistantState)
        
        # Add nodes
        workflow.add_node("classify", self.classify_query)
        workflow.add_node("retrieve", self.retrieve_documentation)
        workflow.add_node("generate", self.generate_bob_ross_response)
        
        # Add edges
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def process_highlighted_text(self, text: str) -> dict:
        """Main entry point for processing highlighted text"""
        try:
            logger.info(f"Processing highlighted text: {text[:100]}...")
            
            # Initialize state
            initial_state = DocumentationAssistantState(
                original_text=text,
                query_type=None,
                context_documents=[],
                bob_ross_response="",
                confidence_score=0.0,
                citations=[],
                processing_steps=["🌟 Starting your happy little journey to understanding..."]
            )
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return {
                "analysis": final_state["bob_ross_response"],
                "original_text": text,
                "query_type": final_state["query_type"],
                "confidence": final_state["confidence_score"],
                "citations": final_state.get("citations", []),
                "processing_steps": final_state.get("processing_steps", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {
                "analysis": "Well, friend, I encountered a happy little accident while processing your request. But remember, we don't make mistakes, just happy little accidents! Please try again.",
                "original_text": text,
                "error": str(e)
            }

# Global agent instance
agent = None

def get_agent():
    """Get or create the global agent instance"""
    global agent
    if agent is None:
        agent = BobRossDocumentationAgent()
    return agent