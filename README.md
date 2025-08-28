# Bob the Helper
## Description
Bob Ross, one of the most helpful and kind people in the world is now stepping up to the easel to help with your support engineering needs. As a Chrome extension, Bob Ross will take whatever you highlight, and give you a headstart cheatsheet on what you need to know and what is now next.

## Requirements
- Langchain acount
- Google Chrome

## Project Structure

```
bob-the-helper/
├── README.md
├── joyOfHighlighting/ (Chrome Extension Frontend)
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── popup.html
│   ├── popup.js
│   └── popup.css
├── happyLittleTreesOfKnowledge/ (Cloud Run Agent)
│   ├── main.py
│   ├── requirements.txt
│   ├── langgraph_agent.py
│   ├── doc_ingestion.py
│   └── evaluation.py
├── evaluation/
│   ├── test_cases.json
│   ├── langsmith_evaluation.py
│   └── results/
└── documentation/
    ├── friction_log.md
    ├── setup_instructions.md
    └── demo_script.md
```
