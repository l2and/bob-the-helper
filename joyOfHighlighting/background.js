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
    
    console.log('🎨 Context menu clicked, selected text:', selectedText);
    
    if (selectedText) {
      try {
        console.log('🚀 Starting LangChain query...');
        const result = await queryLangChain(selectedText);
        
        console.log('✅ Got result from LangChain:', result.substring(0, 200) + '...');
        console.log('📤 Sending showResult message to tab:', tab.id);
        
        chrome.tabs.sendMessage(tab.id, {
          action: 'showResult',
          result: result
        }, (response) => {
          if (chrome.runtime.lastError) {
            console.error('❌ Error sending message to content script:', chrome.runtime.lastError);
          } else {
            console.log('✅ Message sent successfully to content script');
          }
        });
      } catch (error) {
        console.error('❌ Error querying LangChain:', error);
        chrome.tabs.sendMessage(tab.id, {
          action: 'showResult',
          result: `Sorry, there was an error processing your request: ${error.message}`
        }, (response) => {
          if (chrome.runtime.lastError) {
            console.error('❌ Error sending error message to content script:', chrome.runtime.lastError);
          }
        });
      }
    } else {
      console.warn('⚠️ No text selected');
    }
  }
});

async function queryLangChain(text) {
  // Get the LangChain endpoint from storage
  const result = await chrome.storage.sync.get(['langchainEndpoint']);
  let endpoint = result.langchainEndpoint;
  
  // Default to local development endpoint if not configured
  if (!endpoint) {
    endpoint = 'http://localhost:8080/BobRossHelp';
    // Save the default endpoint for future use
    chrome.storage.sync.set({ langchainEndpoint: endpoint });
  }
  
  console.log('Sending request to:', endpoint);
  console.log('Request payload:', { text: text });
  
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
    const errorText = await response.text();
    console.error('HTTP Error Response:', errorText);
    throw new Error(`HTTP error! status: ${response.status}\nDetails: ${errorText}`);
  }
  
  const data = await response.json();
  console.log('Response data:', data);
  
  // Handle the new LangGraph response format
  if (data.analysis) {
    // Add some metadata to the response
    let result = data.analysis;
    
    // Add query type and processing info if available
    if (data.query_type) {
      result = `**Query Type:** ${data.query_type}\n\n${result}`;
    }
    
    if (data.processing_steps && data.processing_steps.length > 0) {
      result += '\n\n---\n**Processing Steps:**\n' + data.processing_steps.join('\n');
    }
    
    return result;
  }
  
  // Fallback for other response formats
  return data.response || data.result || data.output || data.message || 'No response received from Bob Ross Helper';
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'textSelected') {
    // Store the selected text for potential use
    chrome.storage.local.set({ lastSelectedText: message.text });
  }
});