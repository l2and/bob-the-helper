import os
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)
from typing import TypedDict, Literal, List, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langsmith import Client
from logger_config import setup_logger

# Use the same logger configuration as main.py
logger = setup_logger(__name__)

class DocumentationAssistantState(TypedDict):
    """Simple state for Bob Ross Documentation Assistant"""
    original_text: str
    query_type: Optional[str]
    context_info: str
    bob_ross_response: str
    processing_steps: List[str]

class SimpleBobRossAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name="claude-sonnet-4-20250514",
            temperature=0.7,  # Match main.py configuration
            max_tokens=1000,
            timeout=None,
            max_retries=2
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
        
        # Hardcoded documentation - simple but effective for demo
        self.documentation_db = {
            "code_explanation": """
            LangChain provides several key components:
            - LLMChain: A simple chain that combines a prompt template with a language model
            - LCEL (LangChain Expression Language): Use pipe operator (|) to chain components
            - ChatAnthropic: Integration for Claude models with streaming support
            Example: chain = prompt | model | output_parser
            """,
            
            "error_help": """
            Common LangChain errors and solutions:
            - ImportError: Install with 'pip install langchain langchain-anthropic'
            - API Key errors: Set ANTHROPIC_API_KEY environment variable
            - Memory issues: Use text splitters for large documents
            - Rate limits: Implement retry logic and check your API quotas
            """,
            
            "concept_learning": """
            Key LangChain concepts:
            - Chains: Sequences of operations on language models
            - Agents: Systems that use LLMs to choose actions
            - Memory: Ways to persist state between chain calls
            - Retrievers: Components that fetch relevant documents
            - Vector Stores: Databases for semantic similarity search
            """,
            
            "api_usage": """
            LangChain API usage patterns:
            - ChatAnthropic: ChatAnthropic(model_name="claude-sonnet-4-20250514")
            - Streaming: ChatAnthropic(streaming=True)
            - Chains: chain.invoke({"input": "your text"})
            - Async: await chain.ainvoke({"input": "your text"})
            - Batch: chain.batch([{"input": "text1"}, {"input": "text2"}])
            """,
            
            "general_help": """
            LangChain is a framework for building applications with language models.
            It provides tools for prompt management, model integration, memory, 
            retrieval, and agent creation. Start with simple chains and gradually 
            add complexity as needed. The documentation at python.langchain.com 
            has comprehensive guides and examples.
            """
        }
        
        self.graph = self.create_langgraph_workflow()
    
    def classify_query(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Classify the type of help request"""
        try:
            text = state["original_text"]
            steps = state.get("processing_steps", [])
            steps.append("🎨 Analyzing your text to understand what kind of help you need...")
            
            logger.info("📊 Starting text classification...")
            
            classification_prompt = f"""
            Analyze this highlighted text and classify what type of help the user likely needs:
            
            Text: "{text}"
            
            Choose ONE of these categories:
            - code_explanation: User wants to understand what code does
            - error_help: User has an error message they need help with
            - concept_learning: User wants to learn about a concept or term
            - api_usage: User wants to know how to use an API or function
            - general_help: General questions or unclear requests
            
            Look for these clues:
            - Code snippets or imports -> code_explanation
            - Error messages or "ImportError", "AttributeError" -> error_help
            - Questions like "what is", "explain" -> concept_learning
            - Function names, method calls -> api_usage
            - Vague or unclear requests -> general_help
            
            Respond with just the category name.
            """
            
            logger.info("🤖 Sending classification request to Claude...")
            
            # Add simple retry logic for API overload
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    response = self.llm.invoke([HumanMessage(content=classification_prompt)])
                    logger.info("✅ Received classification response from Claude")
                    break
                except Exception as e:
                    if attempt < max_retries and ("overloaded" in str(e).lower() or "529" in str(e)):
                        logger.info(f"⏳ API overloaded (attempt {attempt + 1}/{max_retries + 1}), retrying in 2 seconds...")
                        import time
                        time.sleep(2)
                    else:
                        raise e
            
            query_type = response.content.strip().lower()
            
            # Validate the response
            valid_types = ["code_explanation", "error_help", "concept_learning", "api_usage", "general_help"]
            if query_type not in valid_types:
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
    
    def retrieve_context(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Retrieve context based on query type - using hardcoded docs"""
        try:
            query_type = state["query_type"]
            steps = state.get("processing_steps", [])
            steps.append("🔍 Finding relevant documentation for your question...")
            
            logger.info(f"📚 Starting context retrieval for query type: {query_type}")
            
            # Get context from our hardcoded documentation
            context_info = self.documentation_db.get(query_type, self.documentation_db["general_help"])
            
            logger.info(f"✅ Retrieved context for {query_type}")
            logger.info(f"📄 Context length: {len(context_info)} characters")
            
            return {
                **state,
                "context_info": context_info,
                "processing_steps": steps
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_context: {e}")
            return {
                **state,
                "context_info": "General LangChain help information",
                "processing_steps": steps
            }
    
    def generate_bob_ross_response(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Generate the final Bob Ross-style response"""
        try:
            text = state["original_text"]
            query_type = state["query_type"]
            context = state["context_info"]
            steps = state.get("processing_steps", [])
            steps.append("🎨 Painting you a beautiful explanation with Bob Ross wisdom...")
            
            # Try to use LangSmith prompt first, fallback to local prompt
            if self.bob_ross_prompt:
                try:
                    logger.info("🎨 Using LangSmith prompt: l2and/bob_ross_help")
                    # Create enhanced question with context for LangSmith prompt
                    enhanced_question = f"""
                    User's highlighted text: "{text}"
                    
                    Additional context:
                    - Query type: {query_type}
                    - Relevant documentation: {context}
                    
                    Please help explain this in your warm, encouraging Bob Ross style.
                    """
                    
                    logger.info("🚀 Sending request to LangSmith prompt...")
                    # Chain the LangSmith prompt with the LLM to generate response
                    from langchain_core.output_parsers import StrOutputParser
                    chain = self.bob_ross_prompt | self.llm | StrOutputParser()
                    bob_response = chain.invoke({"question": enhanced_question})
                    logger.info("✅ Successfully used LangSmith prompt")
                    
                except Exception as e:
                    error_str = str(e)
                    if "overloaded" in error_str.lower() or "529" in error_str:
                        logger.warning(f"Claude API temporarily overloaded, falling back to local prompt: {e}")
                    else:
                        logger.warning(f"Failed to use LangSmith prompt: {e}, falling back to local prompt")
                    # Fallback to local prompt
                    fallback_prompt = f"""
                    You are Bob Ross, the gentle painting instructor, now helping people understand LangChain concepts.
                    
                    User's highlighted text: "{text}"
                    Query type: {query_type}
                    Relevant documentation: {context}
                    
                    Respond in Bob Ross's encouraging, gentle style. Use painting metaphors to explain technical concepts.
                    Make complex things feel approachable and reassuring. End with encouragement and next steps.
                    Keep it concise but warm and helpful.
                    """
                    
                    response = self.llm.invoke([HumanMessage(content=fallback_prompt)])
                    bob_response = response.content
            else:
                logger.info("Using fallback prompt (LangSmith prompt not available)")
                # Fallback prompt when LangSmith is not available
                fallback_prompt = f"""
                You are Bob Ross, the gentle painting instructor, now helping people understand LangChain concepts.
                
                User's highlighted text: "{text}"
                Query type: {query_type}
                Relevant documentation: {context}
                
                Respond in Bob Ross's encouraging, gentle style. Use painting metaphors to explain technical concepts.
                Make complex things feel approachable and reassuring. End with encouragement and next steps.
                Keep it concise but warm and helpful.
                """
                
                response = self.llm.invoke([HumanMessage(content=fallback_prompt)])
                bob_response = response.content
            
            logger.info("Generated Bob Ross response successfully")
            
            return {
                **state,
                "bob_ross_response": bob_response,
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
                "processing_steps": steps + ["Had some technical difficulties, but made the best of it!"]
            }
    
    def create_langgraph_workflow(self) -> StateGraph:
        """Create the simple LangGraph workflow"""
        workflow = StateGraph(DocumentationAssistantState)
        
        # Add nodes - this is the key LangGraph demonstration
        workflow.add_node("classify", self.classify_query)
        workflow.add_node("retrieve", self.retrieve_context)
        workflow.add_node("generate", self.generate_bob_ross_response)
        
        # Add edges - showing multi-step workflow
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "generate") 
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def process_highlighted_text(self, text: str) -> dict:
        """Main entry point - demonstrates LangGraph state management"""
        try:
            logger.info(f"Processing highlighted text: {text[:100]}...")
            
            # Initialize state - key LangGraph concept
            initial_state = DocumentationAssistantState(
                original_text=text,
                query_type=None,
                context_info="",
                bob_ross_response="",
                processing_steps=["🌟 Starting your happy little journey to understanding..."]
            )
            
            # Run the graph - this is what makes it LangGraph!
            final_state = self.graph.invoke(initial_state)
            
            return {
                "analysis": final_state["bob_ross_response"],
                "original_text": text,
                "query_type": final_state["query_type"],
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
        agent = SimpleBobRossAgent()
    return agent