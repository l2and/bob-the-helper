import os
import json
from typing import TypedDict, Literal, List, Optional, Dict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

from logger_config import setup_logger

logger = setup_logger(__name__)
# Temporarily enable debug logging to see Claude responses
logger.setLevel(10)  # DEBUG level

class DocumentationAssistantState(TypedDict):
    """Enhanced state with confidence scoring and human-in-the-loop support"""
    original_text: str
    query_type: Optional[str]
    classification_confidence: float
    context_category: Optional[str]
    context_confidence: float
    context_info: str
    bob_ross_response: str
    overall_confidence: float
    processing_steps: List[str]
    confidence_reasons: List[str]
    # Human-in-the-loop fields
    needs_human_input: bool
    human_feedback: Optional[str]  # Additional context from user
    human_classification: Optional[str]  # User-selected classification
    available_classifications: List[Dict[str, str]]  # Options for user to choose from
    session_id: Optional[str]  # For tracking interrupted sessions

class ConfidenceBasedBobRossAgent:
    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    
    def __init__(self):
        self.llm = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-20250514",
            temperature=0.1,  # Lower for classification, higher for generation
            max_tokens=1000,
            timeout=None,
            max_retries=2
        )
        
        # Memory saver for human-in-the-loop checkpointing
        self.memory = MemorySaver()
        
        # Confidence threshold
        self.confidence_threshold = 0.70
        
        # Separate LLM for generation with higher creativity
        self.creative_llm = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-20250514",
            temperature=0.7,  # Higher for Bob Ross creativity
            max_tokens=1500,
            timeout=None,
            max_retries=2
        )
        
        self.langsmith_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        
        # Load the Bob Ross prompt from LangSmith
        try:
            self.bob_ross_prompt = self.langsmith_client.pull_prompt("l2and/bob_ross_help")
            logger.info("Successfully loaded Bob Ross prompt from LangSmith")
        except Exception as e:
            logger.warning(f"Could not load LangSmith prompt: {e}")
            self.bob_ross_prompt = None
        
        # Enhanced documentation with confidence indicators
        self.documentation_db = {
            "code_explanation": {
                "content": """
                LangChain provides several key components:
                - LLMChain: A simple chain that combines a prompt template with a language model
                - LCEL (LangChain Expression Language): Use pipe operator (|) to chain components
                - ChatAnthropic: Integration for Claude models with streaming support
                Example: chain = prompt | model | output_parser
                """,
                "keywords": ["import", "from", "def", "class", "=", "chain", "llm", "prompt"],
                "confidence_boost": 0.2
            },
            
            "error_help": {
                "content": """
                Common LangChain errors and solutions:
                - ImportError: Install with 'pip install langchain langchain-anthropic'
                - API Key errors: Set ANTHROPIC_API_KEY environment variable
                - Memory issues: Use text splitters for large documents
                - Rate limits: Implement retry logic and check your API quotas
                """,
                "keywords": ["error", "exception", "failed", "importerror", "attributeerror", "valueerror"],
                "confidence_boost": 0.3
            },
            
            "concept_learning": {
                "content": """
                Key LangChain concepts:
                - Chains: Sequences of operations on language models
                - Agents: Systems that use LLMs to choose actions
                - Memory: Ways to persist state between chain calls
                - Retrievers: Components that fetch relevant documents
                - Vector Stores: Databases for semantic similarity search
                """,
                "keywords": ["what is", "explain", "concept", "definition", "understand", "learn"],
                "confidence_boost": 0.15
            },
            
            "api_usage": {
                "content": """
                LangChain API usage patterns:
                - ChatAnthropic: ChatAnthropic(model_name="claude-sonnet-4-20250514")
                - Streaming: ChatAnthropic(streaming=True)
                - Chains: chain.invoke({"input": "your text"})
                - Async: await chain.ainvoke({"input": "your text"})
                - Batch: chain.batch([{"input": "text1"}, {"input": "text2"}])
                """,
                "keywords": ["how to", "usage", "invoke", "call", "method", "function", "api"],
                "confidence_boost": 0.25
            },
            
            "general_help": {
                "content": """
                LangChain is a framework for building applications with language models.
                It provides tools for prompt management, model integration, memory, 
                retrieval, and agent creation. Start with simple chains and gradually 
                add complexity as needed. The documentation at python.langchain.com 
                has comprehensive guides and examples.
                """,
                "keywords": [],
                "confidence_boost": 0.0
            }
        }
        
        self.graph = self.create_langgraph_workflow()
    
    # =============================================================================
    # MAIN ENTRY POINT
    # =============================================================================
    
    def process_highlighted_text(self, text: str, session_id: str = None) -> dict:
        """Enhanced main entry point with human-in-the-loop support"""
        try:
            import uuid
            if not session_id:
                session_id = str(uuid.uuid4())
                
            logger.info(f"Processing text with session {session_id}: {text[:100]}...")
            
            # Enhanced initial state with human-in-the-loop fields
            initial_state = DocumentationAssistantState(
                original_text=text,
                query_type=None,
                classification_confidence=0.0,
                context_category=None,
                context_confidence=0.0,
                context_info="",
                bob_ross_response="",
                overall_confidence=0.0,
                processing_steps=["🌟 Starting confidence-based analysis..."],
                confidence_reasons=[],
                # Human-in-the-loop initialization
                needs_human_input=False,
                human_feedback=None,
                human_classification=None,
                available_classifications=[],
                session_id=session_id
            )
            
            # Run the enhanced graph with thread/session support
            config = {"configurable": {"thread_id": session_id}}
            final_state = self.graph.invoke(initial_state, config)
            
            # Debug logging
            logger.info(f"🔍 Final state keys: {list(final_state.keys())}")
            logger.info(f"🔍 needs_human_input in final_state: {final_state.get('needs_human_input')}")
            logger.info(f"🔍 bob_ross_response empty: {not final_state.get('bob_ross_response')}")
            
            # Check if we were interrupted for human input
            # Method 1: Check if needs_human_input flag is set
            # Method 2: Check if workflow didn't complete (no bob_ross_response)
            workflow_interrupted = (
                final_state.get("needs_human_input", False) or 
                not final_state.get("bob_ross_response", "").strip()
            )
            
            if workflow_interrupted and final_state.get("needs_human_input", False):
                logger.info("🚨 Workflow interrupted - human input required")
                interrupt_response = {
                    "status": "human_input_required",
                    "session_id": session_id,
                    "original_text": text,
                    "query_type": final_state.get("query_type"),
                    "classification_confidence": final_state.get("classification_confidence", 0.0),
                    "processing_steps": final_state.get("processing_steps", []),
                    "confidence_reasons": final_state.get("confidence_reasons", []),
                    "available_classifications": final_state.get("available_classifications", []),
                    "message": f"I'm only {final_state.get('classification_confidence', 0.0):.0%} confident about how to help with '{text}'. Could you provide more context?"
                }
                logger.info(f"🔍 About to return interrupt response: {interrupt_response}")
                return interrupt_response
            
            # Normal completion
            return {
                "status": "completed",
                "analysis": final_state["bob_ross_response"],
                "original_text": text,
                "session_id": session_id,
                "query_type": final_state["query_type"],
                "classification_confidence": final_state.get("classification_confidence", 0.0),
                "context_confidence": final_state.get("context_confidence", 0.0),
                "overall_confidence": final_state.get("overall_confidence", 0.0),
                "context_category": final_state.get("context_category", "unknown"),
                "processing_steps": final_state.get("processing_steps", []),
                "confidence_reasons": final_state.get("confidence_reasons", []),
                "confidence_breakdown": {
                    "classification": final_state.get("classification_confidence", 0.0),
                    "context_retrieval": final_state.get("context_confidence", 0.0),
                    "overall": final_state.get("overall_confidence", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced processing: {e}")
            return {
                "status": "error",
                "analysis": "Well friend, I encountered a happy little accident. But that's okay - we learn from these moments! Please try again. 🎨",
                "original_text": text,
                "error": str(e),
                "overall_confidence": 0.1,
                "confidence_reasons": [f"Processing failed: {str(e)}"]
            }
    
    def continue_with_human_input(self, session_id: str, human_feedback: str = None, human_classification: str = None) -> dict:
        """Continue processing after human input"""
        try:
            logger.info(f"Continuing session {session_id} with human input")
            logger.info(f"Human feedback: {human_feedback}")
            logger.info(f"Human classification: {human_classification}")
            
            config = {"configurable": {"thread_id": session_id}}
            
            # Get the current state and inspect it
            state_snapshot = self.graph.get_state(config)
            logger.info(f"🔍 Current state snapshot: {state_snapshot}")
            current_state = state_snapshot.values
            logger.info(f"🔍 Current state keys: {list(current_state.keys())}")
            logger.info(f"🔍 Current state needs_human_input: {current_state.get('needs_human_input')}")
            
            # Update the state with human input
            updated_state = {
                **current_state,
                "human_feedback": human_feedback,
                "human_classification": human_classification,
                "needs_human_input": False  # Allow processing to continue
            }
            logger.info(f"🔍 Updated state with human input, needs_human_input set to False")
            logger.info(f"🔍 About to update state with human_classification: {human_classification}")
            
            # Update the checkpoint state first
            self.graph.update_state(config, updated_state)
            logger.info("🔍 State updated in checkpoint")
            
            # Use stream to continue from checkpoint
            logger.info("🔍 Using stream to continue from updated checkpoint...")
            events = []
            for event in self.graph.stream(None, config):
                logger.info(f"🔍 Stream event: {event}")
                events.append(event)
                
            # Get the final state after streaming
            final_state = self.graph.get_state(config).values
            logger.info(f"🔍 Final state after streaming: {list(final_state.keys())}")
            
            return {
                "status": "completed",
                "analysis": final_state["bob_ross_response"],
                "original_text": current_state["original_text"],
                "session_id": session_id,
                "query_type": final_state["query_type"],
                "classification_confidence": final_state.get("classification_confidence", 0.0),
                "context_confidence": final_state.get("context_confidence", 0.0),
                "overall_confidence": final_state.get("overall_confidence", 0.0),
                "context_category": final_state.get("context_category", "unknown"),
                "processing_steps": final_state.get("processing_steps", []),
                "confidence_reasons": final_state.get("confidence_reasons", []),
                "confidence_breakdown": {
                    "classification": final_state.get("classification_confidence", 0.0),
                    "context_retrieval": final_state.get("context_confidence", 0.0),
                    "overall": final_state.get("overall_confidence", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error continuing with human input: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Sorry, there was an error processing your additional context."
            }
    
    # =============================================================================
    # WORKFLOW GRAPH CREATION
    # =============================================================================
    
    def create_langgraph_workflow(self) -> StateGraph:
        """Create enhanced LangGraph workflow with human-in-the-loop support"""
        workflow = StateGraph(DocumentationAssistantState)
        
        # Add enhanced nodes
        workflow.add_node("classify", self.classify_query_with_confidence)
        workflow.add_node("human_input", self.handle_human_input)  # Human-in-the-loop node
        workflow.add_node("retrieve", self.retrieve_context_with_confidence)
        workflow.add_node("generate", self.generate_confident_response)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Add conditional routing after classification
        workflow.add_conditional_edges(
            "classify",
            self.should_continue_or_interrupt,
            {
                "continue": "retrieve",    # High confidence -> continue to retrieve
                "human_input": "human_input"  # Low confidence -> go to human_input node for interruption
            }
        )
        
        # After human input processing, continue to retrieve
        workflow.add_edge("human_input", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile with memory for checkpointing
        return workflow.compile(checkpointer=self.memory, interrupt_before=["human_input"])
    
    def should_continue_or_interrupt(self, state: DocumentationAssistantState) -> Literal["continue", "human_input"]:
        """Determine if workflow should continue or wait for human input"""
        if state.get("needs_human_input", False):
            logger.info("🚦 Routing to human input node - confidence below threshold")
            return "human_input"
        else:
            logger.info("🚦 Routing to continue processing - confidence above threshold")
            return "continue"
    
    # =============================================================================
    # STEP 1: QUERY CLASSIFICATION
    # =============================================================================
    
    def classify_query_with_confidence(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Enhanced classification with detailed confidence scoring"""
        try:
            text = state["original_text"].lower()
            steps = state.get("processing_steps", [])
            confidence_reasons = []
            
            steps.append("🎨 Analyzing your text with enhanced confidence scoring...")
            
            logger.info("📊 Starting enhanced text classification with confidence analysis...")
            
            # Multi-factor confidence scoring
            classification_prompt = f"""
            Analyze this highlighted text and classify what type of help the user likely needs.
            Provide detailed confidence reasoning.
            
            Text: "{text}"
            
            Choose ONE category and provide confidence score (0.0-1.0):
            - code_explanation: User wants to understand what code does
            - error_help: User has an error message they need help with
            - concept_learning: User wants to learn about a concept or term
            - api_usage: User wants to know how to use an API or function
            - general_help: General questions or unclear requests
            
            Consider these confidence factors:
            1. Text clarity and specificity
            2. Presence of technical keywords
            3. Question structure and intent
            4. Code patterns or error messages
            
            Respond in this exact JSON format:
            {{
                "category": "category_name",
                "confidence": 0.85,
                "reasoning": "Brief explanation of classification decision",
                "confidence_factors": [
                    "Factor 1 that increases/decreases confidence",
                    "Factor 2 that increases/decreases confidence"
                ],
                "ambiguity_notes": "Any ambiguous aspects that reduce confidence"
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=classification_prompt)])
            
            # Debug: Log the raw response
            logger.debug(f"Raw Claude response: {response.content}")
            
            try:
                # Try to extract JSON from response if it's wrapped in markdown or has extra text
                response_text = response.content.strip()
                
                # Look for JSON block in markdown
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    if end != -1:
                        response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    # Generic code block
                    start = response_text.find("```") + 3
                    end = response_text.find("```", start)
                    if end != -1:
                        response_text = response_text[start:end].strip()
                
                # Try to find JSON object boundaries
                if not response_text.startswith("{"):
                    start = response_text.find("{")
                    if start != -1:
                        response_text = response_text[start:]
                
                if not response_text.endswith("}"):
                    end = response_text.rfind("}")
                    if end != -1:
                        response_text = response_text[:end+1]
                
                logger.debug(f"Cleaned JSON text: {response_text}")
                
                result = json.loads(response_text)
                query_type = result.get("category", "general_help")
                classification_confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                confidence_factors = result.get("confidence_factors", [])
                ambiguity_notes = result.get("ambiguity_notes", "")
                
                confidence_reasons.extend(confidence_factors)
                if ambiguity_notes:
                    confidence_reasons.append(f"Ambiguity: {ambiguity_notes}")
                
                logger.info(f"✅ Successfully parsed JSON response: {query_type} (confidence: {classification_confidence})")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.warning(f"Raw response was: {response.content}")
                # Fallback: keyword-based classification with confidence
                query_type, classification_confidence = self._fallback_classification(text)
                confidence_reasons.append("Used fallback keyword-based classification")
            
            # Additional confidence boost based on keyword matching
            if query_type in self.documentation_db:
                keywords = self.documentation_db[query_type]["keywords"]
                keyword_matches = sum(1 for keyword in keywords if keyword in text)
                if keyword_matches > 0:
                    boost = min(0.2, keyword_matches * 0.05)
                    classification_confidence = min(1.0, classification_confidence + boost)
                    confidence_reasons.append(f"Keyword matches boosted confidence by {boost:.2f}")
            
            logger.info(f"Classification: {query_type} (confidence: {classification_confidence:.2f})")
            
            # Check if confidence is below threshold - if so, request human input
            needs_human_input = classification_confidence < self.confidence_threshold
            
            if needs_human_input:
                logger.info(f"🚨 Classification confidence ({classification_confidence:.2f}) below threshold ({self.confidence_threshold})")
                logger.info("🤚 Requesting human input for better classification")
                
                # Prepare classification options for user
                available_classifications = [
                    {"value": "error_help", "label": "🐛 This is an error", "description": "I am seeing an error or exception message"},
                    {"value": "concept_learning", "label": "📖 Explain concept", "description": "I want to understand what this concept or term means"},
                    {"value": "api_usage", "label": "⚙️ Show usage", "description": "I want to know how to use this API or function"},
                    {"value": "code_explanation", "label": "🔍 Code review", "description": "I want to understand what this code does"},
                    {"value": "implementation_help", "label": "🚀 Implementation help", "description": "I want help implementing or building something"}
                ]
                
                steps.append("🤚 Low confidence detected - requesting human guidance...")
                confidence_reasons.append(f"Classification confidence {classification_confidence:.2f} is below threshold {self.confidence_threshold}")
            else:
                available_classifications = []
            
            return {
                **state,
                "query_type": query_type,
                "classification_confidence": classification_confidence,
                "processing_steps": steps,
                "confidence_reasons": confidence_reasons,
                "needs_human_input": needs_human_input,
                "available_classifications": available_classifications
            }
            
        except Exception as e:
            logger.error(f"Error in classify_query_with_confidence: {e}")
            return {
                **state,
                "query_type": "general_help",
                "classification_confidence": 0.3,
                "processing_steps": steps + ["Had trouble with classification, using general approach"],
                "confidence_reasons": [f"Classification failed: {str(e)}"]
            }
    
    def _fallback_classification(self, text: str) -> tuple[str, float]:
        """Fallback keyword-based classification with confidence"""
        text_lower = text.lower()
        
        # Define keyword patterns with confidence scores
        patterns = [
            (["import", "from", "def", "class", "=", "()"], "code_explanation", 0.8),
            (["error", "exception", "failed", "importerror"], "error_help", 0.9),
            (["what is", "explain", "definition", "concept"], "concept_learning", 0.7),
            (["how to", "usage", "invoke", "method"], "api_usage", 0.7)
        ]
        
        best_match = ("general_help", 0.4)
        
        for keywords, category, base_confidence in patterns:
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Confidence increases with more matches
                confidence = min(1.0, base_confidence + (matches - 1) * 0.1)
                if confidence > best_match[1]:
                    best_match = (category, confidence)
        
        return best_match
    
    # =============================================================================
    # STEP 2: HUMAN-IN-THE-LOOP PROCESSING
    # =============================================================================
    
    def handle_human_input(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Process human input and continue with refined classification"""
        try:
            logger.info("🤚 ===== ENTERED handle_human_input node =====")
            logger.info(f"🤚 State keys: {list(state.keys())}")
            logger.info(f"🤚 human_classification in state: {state.get('human_classification')}")
            logger.info(f"🤚 human_feedback in state: {state.get('human_feedback')}")
            logger.info(f"🤚 Original query_type: {state.get('query_type')}")
            
            steps = state.get("processing_steps", [])
            confidence_reasons = state.get("confidence_reasons", [])
            
            # If human provided classification, use it
            if state.get("human_classification"):
                query_type = state["human_classification"]
                classification_confidence = 0.9  # High confidence since human selected
                steps.append(f"✅ Human selected classification: {query_type}")
                confidence_reasons.append("Human provided explicit classification")
                logger.info(f"🤚 ✅ OVERRIDING with human classification: {query_type}")
            else:
                # Keep original classification but boost confidence due to human context
                query_type = state.get("query_type", "general_help")
                classification_confidence = min(0.9, state.get("classification_confidence", 0.5) + 0.3)
                steps.append("✅ Human provided additional context")
                confidence_reasons.append("Human provided additional context to improve classification")
                logger.info(f"🤚 Human provided context, boosted confidence to {classification_confidence:.2f}")
            
            # If human provided feedback, incorporate it into the original text
            enhanced_text = state["original_text"]
            if state.get("human_feedback"):
                enhanced_text = f"{state['original_text']}\n\nAdditional context: {state['human_feedback']}"
                steps.append("📝 Incorporated human feedback into analysis")
                logger.info(f"📝 Enhanced text with human feedback: {state['human_feedback'][:100]}...")
            
            return {
                **state,
                "original_text": enhanced_text,
                "query_type": query_type,
                "classification_confidence": classification_confidence,
                "processing_steps": steps,
                "confidence_reasons": confidence_reasons,
                "needs_human_input": False  # Clear the flag to continue processing
            }
            
        except Exception as e:
            logger.error(f"Error in handle_human_input: {e}")
            return {
                **state,
                "needs_human_input": False,
                "processing_steps": state.get("processing_steps", []) + ["Error processing human input, continuing..."]
            }
    
    # =============================================================================
    # STEP 3: CONTEXT RETRIEVAL
    # =============================================================================
    
    def retrieve_context_with_confidence(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Enhanced context retrieval with confidence scoring"""
        try:
            text = state["original_text"]
            query_type = state["query_type"]
            steps = state.get("processing_steps", [])
            confidence_reasons = state.get("confidence_reasons", [])
            
            steps.append("🔍 Searching documentation with confidence assessment...")
            
            logger.info(f"📚 Enhanced context retrieval for: {query_type}")
            
            # Get context from documentation database
            doc_entry = self.documentation_db.get(query_type, self.documentation_db["general_help"])
            context_info = doc_entry["content"]
            
            # Calculate context confidence
            context_confidence = 0.5  # Base confidence
            
            # Boost confidence if we have specific documentation for this category
            if query_type != "general_help":
                context_confidence += 0.3
                confidence_reasons.append("Found specific documentation category match")
            
            # Check keyword relevance in context
            keywords = doc_entry.get("keywords", [])
            if keywords:
                text_lower = text.lower()
                keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
                if keyword_matches > 0:
                    keyword_boost = min(0.2, keyword_matches * 0.05)
                    context_confidence = min(1.0, context_confidence + keyword_boost)
                    confidence_reasons.append(f"Context keywords matched user text ({keyword_matches} matches)")
            
            # Determine context category for better evaluation
            context_category = self._determine_context_category(text, query_type)
            
            logger.info(f"Context confidence: {context_confidence:.2f}, Category: {context_category}")
            
            return {
                **state,
                "context_info": context_info,
                "context_confidence": context_confidence,
                "context_category": context_category,
                "processing_steps": steps,
                "confidence_reasons": confidence_reasons
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_context_with_confidence: {e}")
            return {
                **state,
                "context_info": "General LangChain information",
                "context_confidence": 0.2,
                "context_category": "fallback",
                "processing_steps": steps,
                "confidence_reasons": confidence_reasons + [f"Context retrieval failed: {str(e)}"]
            }
    
    def _determine_context_category(self, text: str, query_type: str) -> str:
        """Determine the specific category of context we're providing"""
        text_lower = text.lower()
        
        if "langgraph" in text_lower or "graph" in text_lower:
            return "langgraph_specific"
        elif "lcel" in text_lower or "pipe" in text_lower or "|" in text:
            return "lcel_specific"
        elif "vector" in text_lower or "embedding" in text_lower:
            return "vector_store_specific"
        elif "chain" in text_lower:
            return "chain_specific"
        elif "agent" in text_lower:
            return "agent_specific"
        elif query_type == "error_help":
            return "troubleshooting_specific"
        else:
            return "general_langchain"
    
    # =============================================================================
    # STEP 4: RESPONSE GENERATION
    # =============================================================================
    
    def generate_confident_response(self, state: DocumentationAssistantState) -> DocumentationAssistantState:
        """Generate response with overall confidence calculation"""
        try:
            text = state["original_text"]
            # Use human classification if provided, otherwise use original query_type
            query_type = state.get("human_classification") or state["query_type"]
            context = state["context_info"]
            
            logger.info(f"🎨 Generating response for query_type: {query_type}")
            if state.get("human_classification"):
                logger.info(f"🎨 Using human-provided classification: {state['human_classification']}")
            else:
                logger.info(f"🎨 Using original classification: {state['query_type']}")
            classification_confidence = state.get("classification_confidence", 0.5)
            context_confidence = state.get("context_confidence", 0.5)
            steps = state.get("processing_steps", [])
            confidence_reasons = state.get("confidence_reasons", [])
            
            steps.append("🎨 Painting your response with calculated confidence...")
            
            # Calculate overall confidence (weighted average)
            overall_confidence = (
                classification_confidence * 0.4 +  # 40% weight on classification
                context_confidence * 0.6           # 60% weight on context quality
            )
            
            confidence_reasons.append(f"Overall confidence: {overall_confidence:.2f} (classification: {classification_confidence:.2f}, context: {context_confidence:.2f})")
            
            # Use creative LLM for response generation
            if self.bob_ross_prompt:
                try:
                    logger.info("🎨 Using LangSmith prompt with confidence context")
                    enhanced_question = f"""
                    User's highlighted text: "{text}"
                    Query type: {query_type} (confidence: {classification_confidence:.2f})
                    Context quality: {context_confidence:.2f}
                    Overall confidence: {overall_confidence:.2f}
                    
                    Relevant documentation: {context}
                    
                    Please provide a Bob Ross-style explanation. If confidence is low (<0.6), 
                    acknowledge uncertainty gently and ask for clarification.
                    """
                    
                    from langchain_core.output_parsers import StrOutputParser
                    chain = self.bob_ross_prompt | self.creative_llm | StrOutputParser()
                    bob_response = chain.invoke({"question": enhanced_question})
                    
                except Exception as e:
                    logger.warning(f"LangSmith prompt failed, using fallback: {e}")
                    bob_response = self._generate_fallback_response(text, query_type, context, overall_confidence)
            else:
                bob_response = self._generate_fallback_response(text, query_type, context, overall_confidence)
            
            # Add confidence indicator to response if low
            if overall_confidence < 0.6:
                bob_response += f"\n\n*Note: I'm about {overall_confidence:.0%} confident in this explanation. Feel free to provide more context if you'd like a more targeted response!*"
            
            logger.info(f"Generated response with overall confidence: {overall_confidence:.2f}")
            
            return {
                **state,
                "bob_ross_response": bob_response,
                "overall_confidence": overall_confidence,
                "processing_steps": steps + ["✨ Your confidence-scored explanation is ready!"],
                "confidence_reasons": confidence_reasons
            }
            
        except Exception as e:
            logger.error(f"Error in generate_confident_response: {e}")
            fallback_response = """
            Well hello there, friend! I see you've highlighted some text, and that's just wonderful. 
            I'm having a little technical difficulty right now, but remember - every mistake is just 
            a happy little accident. Let's keep learning together! 🎨
            """
            
            return {
                **state,
                "bob_ross_response": fallback_response,
                "overall_confidence": 0.2,
                "processing_steps": steps + ["Had technical difficulties but kept a positive attitude!"],
                "confidence_reasons": confidence_reasons + [f"Response generation failed: {str(e)}"]
            }
    
    def _generate_fallback_response(self, text: str, query_type: str, context: str, confidence: float) -> str:
        """Generate fallback response with confidence awareness"""
        confidence_modifier = ""
        if confidence < 0.5:
            confidence_modifier = "I think this might be related to what you're asking about, though I'm not entirely certain. "
        elif confidence > 0.8:
            confidence_modifier = "I'm quite confident this is exactly what you need to know! "
        
        fallback_prompt = f"""
        You are Bob Ross, the gentle painting instructor, helping with LangChain concepts.
        
        User's text: "{text}"
        Query type: {query_type}
        Your confidence level: {confidence:.2f}
        Context: {context}
        
        {confidence_modifier}Respond in Bob Ross's encouraging style, using painting metaphors.
        Keep it concise but warm. If confidence is low, gently acknowledge uncertainty.
        """
        
        response = self.creative_llm.invoke([HumanMessage(content=fallback_prompt)])
        return response.content
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def should_continue_or_interrupt(self, state: DocumentationAssistantState) -> Literal["continue", "human_input"]:
        """Determine if workflow should continue or wait for human input"""
        if state.get("needs_human_input", False):
            logger.info("🚦 Routing to human input node - confidence below threshold")
            return "human_input"
        else:
            logger.info("🚦 Routing to continue processing - confidence above threshold")
            return "continue"
    
    def create_langgraph_workflow(self) -> StateGraph:
        """Create enhanced LangGraph workflow with human-in-the-loop support"""
        workflow = StateGraph(DocumentationAssistantState)
        
        # Add enhanced nodes
        workflow.add_node("classify", self.classify_query_with_confidence)
        workflow.add_node("human_input", self.handle_human_input)  # Human-in-the-loop node
        workflow.add_node("retrieve", self.retrieve_context_with_confidence)
        workflow.add_node("generate", self.generate_confident_response)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Add conditional routing after classification
        workflow.add_conditional_edges(
            "classify",
            self.should_continue_or_interrupt,
            {
                "continue": "retrieve",    # High confidence -> continue to retrieve
                "human_input": "human_input"  # Low confidence -> go to human_input node for interruption
            }
        )
        
        # After human input processing, continue to retrieve
        workflow.add_edge("human_input", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile with memory for checkpointing
        return workflow.compile(checkpointer=self.memory, interrupt_before=["human_input"])
    
    def process_highlighted_text(self, text: str, session_id: str = None) -> dict:
        """Enhanced main entry point with human-in-the-loop support"""
        try:
            import uuid
            if not session_id:
                session_id = str(uuid.uuid4())
                
            logger.info(f"Processing text with session {session_id}: {text[:100]}...")
            
            # Enhanced initial state with human-in-the-loop fields
            initial_state = DocumentationAssistantState(
                original_text=text,
                query_type=None,
                classification_confidence=0.0,
                context_category=None,
                context_confidence=0.0,
                context_info="",
                bob_ross_response="",
                overall_confidence=0.0,
                processing_steps=["🌟 Starting confidence-based analysis..."],
                confidence_reasons=[],
                # Human-in-the-loop initialization
                needs_human_input=False,
                human_feedback=None,
                human_classification=None,
                available_classifications=[],
                session_id=session_id
            )
            
            # Run the enhanced graph with thread/session support
            config = {"configurable": {"thread_id": session_id}}
            final_state = self.graph.invoke(initial_state, config)
            
            # Debug logging
            logger.info(f"🔍 Final state keys: {list(final_state.keys())}")
            logger.info(f"🔍 needs_human_input in final_state: {final_state.get('needs_human_input')}")
            logger.info(f"🔍 bob_ross_response empty: {not final_state.get('bob_ross_response')}")
            
            # Check if we were interrupted for human input
            # Method 1: Check if needs_human_input flag is set
            # Method 2: Check if workflow didn't complete (no bob_ross_response)
            workflow_interrupted = (
                final_state.get("needs_human_input", False) or 
                not final_state.get("bob_ross_response", "").strip()
            )
            
            if workflow_interrupted and final_state.get("needs_human_input", False):
                logger.info("🚨 Workflow interrupted - human input required")
                interrupt_response = {
                    "status": "human_input_required",
                    "session_id": session_id,
                    "original_text": text,
                    "query_type": final_state.get("query_type"),
                    "classification_confidence": final_state.get("classification_confidence", 0.0),
                    "processing_steps": final_state.get("processing_steps", []),
                    "confidence_reasons": final_state.get("confidence_reasons", []),
                    "available_classifications": final_state.get("available_classifications", []),
                    "message": f"I'm only {final_state.get('classification_confidence', 0.0):.0%} confident about how to help with '{text}'. Could you provide more context?"
                }
                logger.info(f"🔍 About to return interrupt response: {interrupt_response}")
                return interrupt_response
            
            # Normal completion
            return {
                "status": "completed",
                "analysis": final_state["bob_ross_response"],
                "original_text": text,
                "session_id": session_id,
                "query_type": final_state["query_type"],
                "classification_confidence": final_state.get("classification_confidence", 0.0),
                "context_confidence": final_state.get("context_confidence", 0.0),
                "overall_confidence": final_state.get("overall_confidence", 0.0),
                "context_category": final_state.get("context_category", "unknown"),
                "processing_steps": final_state.get("processing_steps", []),
                "confidence_reasons": final_state.get("confidence_reasons", []),
                "confidence_breakdown": {
                    "classification": final_state.get("classification_confidence", 0.0),
                    "context_retrieval": final_state.get("context_confidence", 0.0),
                    "overall": final_state.get("overall_confidence", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced processing: {e}")
            return {
                "status": "error",
                "analysis": "Well friend, I encountered a happy little accident. But that's okay - we learn from these moments! Please try again. 🎨",
                "original_text": text,
                "error": str(e),
                "overall_confidence": 0.1,
                "confidence_reasons": [f"Processing failed: {str(e)}"]
            }
    
    def continue_with_human_input(self, session_id: str, human_feedback: str = None, human_classification: str = None) -> dict:
        """Continue processing after human input"""
        try:
            logger.info(f"Continuing session {session_id} with human input")
            logger.info(f"Human feedback: {human_feedback}")
            logger.info(f"Human classification: {human_classification}")
            
            config = {"configurable": {"thread_id": session_id}}
            
            # Get the current state and inspect it
            state_snapshot = self.graph.get_state(config)
            logger.info(f"🔍 Current state snapshot: {state_snapshot}")
            current_state = state_snapshot.values
            logger.info(f"🔍 Current state keys: {list(current_state.keys())}")
            logger.info(f"🔍 Current state needs_human_input: {current_state.get('needs_human_input')}")
            
            # Update the state with human input
            updated_state = {
                **current_state,
                "human_feedback": human_feedback,
                "human_classification": human_classification,
                "needs_human_input": False  # Allow processing to continue
            }
            logger.info(f"🔍 Updated state with human input, needs_human_input set to False")
            logger.info(f"🔍 About to update state with human_classification: {human_classification}")
            
            # Update the checkpoint state first
            self.graph.update_state(config, updated_state)
            logger.info("🔍 State updated in checkpoint")
            
            # Use stream to continue from checkpoint
            logger.info("🔍 Using stream to continue from updated checkpoint...")
            events = []
            for event in self.graph.stream(None, config):
                logger.info(f"🔍 Stream event: {event}")
                events.append(event)
                
            # Get the final state after streaming
            final_state = self.graph.get_state(config).values
            logger.info(f"🔍 Final state after streaming: {list(final_state.keys())}")
            
            return {
                "status": "completed",
                "analysis": final_state["bob_ross_response"],
                "original_text": current_state["original_text"],
                "session_id": session_id,
                "query_type": final_state["query_type"],
                "classification_confidence": final_state.get("classification_confidence", 0.0),
                "context_confidence": final_state.get("context_confidence", 0.0),
                "overall_confidence": final_state.get("overall_confidence", 0.0),
                "context_category": final_state.get("context_category", "unknown"),
                "processing_steps": final_state.get("processing_steps", []),
                "confidence_reasons": final_state.get("confidence_reasons", []),
                "confidence_breakdown": {
                    "classification": final_state.get("classification_confidence", 0.0),
                    "context_retrieval": final_state.get("context_confidence", 0.0),
                    "overall": final_state.get("overall_confidence", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error continuing with human input: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Sorry, there was an error processing your additional context."
            }

# Global agent instance
agent = None

def get_agent():
    """Get or create the enhanced global agent instance"""
    global agent
    if agent is None:
        agent = ConfidenceBasedBobRossAgent()
    return agent