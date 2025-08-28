# Bob Ross Helper Chrome Extension

A Chrome extension that lets you highlight text and get Bob Ross-style explanations via your LangChain instance.

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select this directory
4. The extension should now appear in your extensions list

## Setup

1. Click the extension icon in your Chrome toolbar
2. Enter your LangChain endpoint URL (e.g., `https://your-cloudbuild-instance.com/api`)
3. Click "Save Configuration"
4. Optionally click "Test Connection" to verify it's working

## Usage

1. Highlight any text on any webpage
2. Right-click to open the context menu
3. Select "Bob Ross, HELP! 🎨"
4. A modal will appear with Bob Ross-style explanation of the highlighted text
5. The modal automatically closes after 30 seconds or when you click the X

## LangChain API Requirements

Your LangChain endpoint should accept POST requests with this format:
```json
{
  "prompt": "Your Bob Ross style prompt...",
  "text": "The highlighted text"
}
```

And return a response with one of these fields:
- `response`
- `result` 
- `output`

## Files

- `manifest.json` - Extension configuration
- `background.js` - Service worker for context menu and API calls
- `content.js` - Content script for text selection and result display
- `popup.html/js` - Configuration interface