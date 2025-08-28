chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'bob-ross-help',
    title: 'Bob Ross, HELP! 🎨',
    contexts: ['selection']
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'bob-ross-help') {
    const selectedText = info.selectionText;
    
    if (selectedText) {
      try {
        const result = await queryLangChain(selectedText);
        
        chrome.tabs.sendMessage(tab.id, {
          action: 'showResult',
          result: result
        });
      } catch (error) {
        console.error('Error querying LangChain:', error);
        chrome.tabs.sendMessage(tab.id, {
          action: 'showResult',
          result: 'Sorry, there was an error processing your request. Please check your LangChain endpoint configuration.'
        });
      }
    }
  }
});

async function queryLangChain(text) {
  // Get the LangChain endpoint from storage
  const result = await chrome.storage.sync.get(['langchainEndpoint']);
  const endpoint = result.langchainEndpoint;
  
  if (!endpoint) {
    throw new Error('LangChain endpoint not configured. Please set it in the extension popup.');
  }
  
  // Send request in format expected by Flask app
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text
    })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  const data = await response.json();
  return data.analysis || data.response || data.result || data.output || 'No response from LangChain';
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'textSelected') {
    // Store the selected text for potential use
    chrome.storage.local.set({ lastSelectedText: message.text });
  }
});