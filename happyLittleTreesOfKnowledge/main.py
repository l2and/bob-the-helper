# main.py
import os
from flask import Flask, jsonify, request
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
prompt = client.pull_prompt("l2and/bob_ross_help", include_model=True)

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        "message": "Bob Ross is ready to help with LangChain + Claude",
        "endpoints": ["/BobRossHelp"],
        "llm": "Claude (Anthropic)"
    })

@app.route('/BobRossHelp', methods=['POST'])
def BobRossHelp():
    try:
        data = request.get_json()
        selected_text = data.get('text', '')
        
        if not selected_text:
            return jsonify({"error": "No text provided"}), 400
        
        # Use your LangSmith prompt
        llm = ChatAnthropic(
            anthropic_api_key=os.getenv("LANGSMITH_API_KEY"),
            model_name="claude-sonnet-4-20250514"
        )
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({"text": selected_text})
        
        return jsonify({
            "analysis": result,
            "original_text": selected_text
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Error processing request"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))