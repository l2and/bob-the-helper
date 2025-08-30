# main.py
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv

# Load environment variables from .env file in project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
print(f"🔧 Loading environment variables from: {dotenv_path}")
load_dotenv(dotenv_path)
from langgraph_agent import get_agent
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, Callable
from logger_config import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)
# Enable CORS for Chrome extension with explicit configuration
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], 
     allow_headers=["Content-Type", "Authorization"])


def run_with_timeout(func: Callable, timeout_seconds: int = 30, *args, **kwargs):
    """Run a function with a timeout"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

# Initialize LangSmith client
try:
    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    logger.info("LangSmith client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LangSmith client: {e}")
    client = None

# Initialize the LLM following LangChain best practices
def create_llm():
    """Create ChatAnthropic instance with proper error handling"""
    try:
        # Rebuild the model to fix Pydantic v2 compatibility issues
        ChatAnthropic.model_rebuild()
        
        return ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model_name="claude-sonnet-4-20250514",
            temperature=0.7,  # A bit of creativity for Bob Ross
            max_tokens=1000,
            timeout=None,  
            max_retries=2
        )
    except Exception as e:
        logger.error(f"Failed to create LLM: {e}")
        raise

# Create a fallback prompt if LangSmith is unavailable
FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are Bob Ross, the legendary painter and art instructor, but now you're helping people understand code, technology, and documentation. 

Respond to questions about the highlighted text in your characteristic warm, encouraging, and gentle style. Use painting metaphors where appropriate, but focus on being genuinely helpful.

Key principles:
- Be encouraging and positive
- Use gentle, soothing language
- Explain complex concepts in simple terms
- Include relevant technical details
- End with encouragement

Remember: "There are no mistakes, only happy little learning opportunities!" """),
    ("human", "Please help me understand this text: {question}")
])

@app.route('/')
def hello():
    """Health check endpoint"""
    return jsonify({
        "message": "🎨 Bob Ross is ready to help with LangChain + Claude",
        "status": "healthy",
        "endpoints": ["/BobRossHelp", "/health"],
        "llm": "Claude Sonnet 4",
        "langsmith_connected": client is not None
    })

@app.route('/health')
def health_check():
    """Detailed health check for monitoring"""
    logger.info("🏥 Starting comprehensive health check")
    
    logger.info("📊 Initializing health status structure")
    health_status = {
        "status": "healthy",
        "timestamp": os.popen("date").read().strip(),
        "environment": os.getenv("FLASK_ENV", "production")
    }
    logger.info(f"📊 Basic health status initialized: {health_status}")
    
    # LangSmith detailed status
    logger.info("🔍 Beginning LangSmith connectivity assessment")
    langsmith_status = {
        "connected": client is not None,
        "api_key_configured": bool(os.getenv("LANGSMITH_API_KEY"))
    }
    logger.info(f"🔍 LangSmith initial status: connected={langsmith_status['connected']}, api_key_configured={langsmith_status['api_key_configured']}")
    
    if client:
        # Test prompt access - simplified without complex timeout
        logger.info("🔗 Testing LangSmith prompt access...")
        try:
            test_prompt = client.pull_prompt("l2and/bob_ross_help", include_model=True)
            langsmith_status["prompt_accessible"] = True
            langsmith_status["prompt_name"] = "l2and/bob_ross_help"
            langsmith_status["prompt_type"] = type(test_prompt).__name__
            logger.info("✅ LangSmith prompt test successful")
        except Exception as e:
            langsmith_status["prompt_accessible"] = False
            langsmith_status["prompt_error"] = str(e)
            logger.warning(f"❌ LangSmith prompt test failed: {e}")
        
        # Test API connectivity - simplified without complex timeout
        logger.info("🌐 Testing LangSmith API connection...")
        try:
            list(client.list_runs(limit=1))  # Convert to list to force execution
            langsmith_status["api_connection"] = "successful"
            logger.info("✅ LangSmith API connection test successful")
        except Exception as e:
            langsmith_status["api_connection"] = "failed"
            langsmith_status["api_error"] = str(e)
            logger.warning(f"❌ LangSmith API test failed: {e}")
    else:
        langsmith_status["reason"] = "API key not configured or client initialization failed"
    
    health_status["langsmith"] = langsmith_status
    
    # Claude/Anthropic detailed status
    claude_status = {
        "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "model": "claude-sonnet-4-20250514"
    }
    
    try:
        # Test Claude LLM creation - simplified without complex timeout
        logger.info("🤖 Testing Claude LLM connectivity...")
        llm = create_llm()
        claude_status["llm_accessible"] = True
        claude_status["temperature"] = 0.7
        claude_status["max_tokens"] = 1000
        logger.info("✅ Claude LLM initialized successfully")
        
        # Test a simple API call - simplified without complex timeout
        logger.info("🔗 Testing Claude API with simple prompt...")
        test_chain = FALLBACK_PROMPT | llm | StrOutputParser()
        test_result = test_chain.invoke({"question": "test"})
        claude_status["api_test"] = "successful"
        claude_status["test_response_length"] = len(test_result)
        logger.info("✅ Claude API test successful")
        
    except Exception as e:
        claude_status["llm_accessible"] = False
        claude_status["error"] = str(e)
        health_status["status"] = "unhealthy"
        logger.error(f"❌ Claude health check failed: {e}")
    
    health_status["claude"] = claude_status
    
    # Current configuration
    health_status["current_config"] = {
        "prompt_source": "fallback" if not client else "attempting_langsmith",
        "fallback_prompt_configured": FALLBACK_PROMPT is not None,
        "cors_enabled": True
    }
    
    logger.info(f"Health check completed with status: {health_status['status']}")
    return jsonify(health_status)

@app.route('/BobRossHelp', methods=['POST'])
def bob_ross_help():
    """Main endpoint - now powered by LangGraph agent"""
    try:
        logger.info("🎨 Received Bob Ross help request")
        
        # Validate request (keep your existing validation logic)
        logger.info("Validating incoming request format")
        if not request.is_json:
            logger.warning("Request rejected: Content-Type is not application/json")
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        selected_text = data.get('text', '').strip()
        
        if not selected_text:
            logger.warning("Request rejected: No text provided in request body")
            return jsonify({
                "error": "No text provided",
                "message": "Please highlight some text first, friend!"
            }), 400
        
        text_preview = selected_text[:100] + "..." if len(selected_text) > 100 else selected_text
        logger.info(f"📝 Processing text ({len(selected_text)} chars): {text_preview}")
        
        # NEW: Initialize LangGraph agent instead of LLM + chain
        logger.info("🤖 Initializing LangGraph agent")
        agent = get_agent()
        logger.info("✅ LangGraph agent initialized successfully")
        
        # NEW: Process with LangGraph workflow
        logger.info("🔗 Starting LangGraph multi-step workflow")
        logger.info("✅ Workflow: Classify → Retrieve → Generate")
        
        # Invoke the agent
        logger.info("🚀 Sending request to LangGraph agent")
        result = agent.process_highlighted_text(selected_text)
        
        # Extract the response and log details
        analysis = result.get("analysis", "")
        query_type = result.get("query_type", "unknown")
        processing_steps = result.get("processing_steps", [])
        
        logger.info(f"✅ Received response from LangGraph agent ({len(analysis)} chars)")
        logger.info(f"🎯 Classified as: {query_type}")
        logger.info(f"📊 Processing steps: {len(processing_steps)}")
        
        response_preview = analysis[:300] + "..." if len(analysis) > 150 else analysis
        logger.info(f"📤 Response preview: {response_preview}")
        
        logger.info("🎉 LangGraph workflow completed successfully")
        
        return jsonify({
            "analysis": analysis,
            "original_text": selected_text,
            "status": "success",
            "query_type": query_type,
            "processing_steps": processing_steps,
            "source": "langgraph_agent",
            "processing_info": {
                "text_length": len(selected_text),
                "response_length": len(analysis),
                "workflow": "classify → retrieve → generate",
                "agent_type": "LangGraph Multi-Step Agent"
            }
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "message": "Oops! Even happy little accidents happen. Please try again.",
            "status": "error"
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Custom 404 handler"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "This isn't the canvas you're looking for, friend!",
        "available_endpoints": ["/", "/BobRossHelp", "/health"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "Even Bob Ross has off days. Please try again!"
    }), 500

if __name__ == '__main__':
    # Enhanced debugging configuration
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    port = int(os.environ.get('PORT', 8080))
    
    # Check API keys and show detailed status
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    langsmith_key = os.getenv('LANGSMITH_API_KEY')
    
    logger.info("🎨" + "="*50)
    logger.info("🎨 Bob Ross Helper Starting Up")
    logger.info("🎨" + "="*50)
    logger.info(f"🔧 Environment file loaded from: {dotenv_path}")
    logger.info(f"🔧 Environment file exists: {'✅' if os.path.exists(dotenv_path) else '❌'}")
    logger.info(f"🌐 Server: http://0.0.0.0:{port}")
    logger.info(f"🔧 Debug Mode: {debug_mode}")
    
    # More detailed API key logging
    if anthropic_key:
        logger.info(f"🔑 Anthropic API: ✅ Configured (key: ...{anthropic_key[-8:]})")
    else:
        logger.error("🔑 Anthropic API: ❌ Missing - Check ANTHROPIC_API_KEY in .env file")
    
    if langsmith_key:
        logger.info(f"🔑 LangSmith API: ✅ Configured (key: ...{langsmith_key[-8:]})")
    else:
        logger.warning("🔑 LangSmith API: ❌ Missing - Check LANGSMITH_API_KEY in .env file")
    
    # Determine which prompt source will be used
    if client and langsmith_key:
        try:
            logger.info("📋 Testing LangSmith prompt accessibility for startup")
            test_prompt = client.pull_prompt("l2and/bob_ross_help", include_model=True)
            logger.info("📋 Prompt Source: ✅ LangSmith (l2and/bob_ross_help) - Available")
        except Exception as e:
            logger.warning(f"📋 Prompt Source: ⚠️ LangSmith failed, will use Fallback - {e}")
    else:
        logger.info("📋 Prompt Source: 📝 Fallback (Local template)")
    
    logger.info("🎨" + "="*50)
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode,
        use_reloader=debug_mode
    )