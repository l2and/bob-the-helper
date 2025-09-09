chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'bob-ross-help',
    title: 'Bob Ross, HELP! 🎨',
    contexts: ['selection']
  });
  
  // Add settings context menu for extension icon
  chrome.contextMenus.create({
    id: 'bob-ross-settings',
    title: 'Settings',
    contexts: ['action']
  });
});

// Handle toolbar icon clicks to toggle side panel
chrome.action.onClicked.addListener(async (tab) => {
  console.log('🎨 Toolbar icon clicked, toggling side panel');
  try {
    await chrome.sidePanel.open({ tabId: tab.id });
  } catch (error) {
    console.error('❌ Error opening side panel:', error);
  }
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'bob-ross-help') {
    const selectedText = info.selectionText;
    
    console.log('🎨 Context menu clicked, selected text:', selectedText);
    
    if (selectedText) {
      try {
        console.log('🚀 Starting LangChain query...');
        
        // Open side panel and show progress
        await chrome.sidePanel.open({ tabId: tab.id });
        chrome.runtime.sendMessage({
          target: 'sidepanel',
          action: 'showProgress'
        });
        
        const responseData = await queryLangChain(selectedText);
        
        console.log('✅ Got result from LangChain:', responseData);
        console.log('📤 Sending showResult message to sidepanel');
        
        // Send to side panel instead of content script
        chrome.runtime.sendMessage({
          target: 'sidepanel',
          action: 'showResult',
          result: responseData.analysis || responseData,
          fullResponse: responseData,
          originalText: selectedText
        });
      } catch (error) {
        console.error('❌ Error querying LangChain:', error);
        
        // Open side panel even for errors
        await chrome.sidePanel.open({ tabId: tab.id });
        
        chrome.runtime.sendMessage({
          target: 'sidepanel',
          action: 'showResult',
          result: `Sorry, there was an error processing your request: ${error.message}`,
          fullResponse: { error: error.message, overall_confidence: 0.0 }
        });
      }
    } else {
      console.warn('⚠️ No text selected');
    }
  } else if (info.menuItemId === 'bob-ross-settings') {
    // Open settings popup
    console.log('⚙️ Settings menu clicked');
    try {
      await chrome.windows.create({
        url: 'popup.html',
        type: 'popup',
        width: 400,
        height: 500,
        focused: true
      });
    } catch (error) {
      console.error('❌ Error opening settings popup:', error);
    }
  }
});

async function queryLangChain(text) {
  // Get the LangChain endpoint from storage
  const result = await chrome.storage.sync.get(['langchainEndpoint']);
  let endpoint = result.langchainEndpoint;
  
  // Default to local development endpoint if not configured
  if (!endpoint) {
    endpoint = 'http://127.0.0.1:8080/BobRossHelp';
    // Save the default endpoint for future use
    chrome.storage.sync.set({ langchainEndpoint: endpoint });
  }
  
  console.log('🌐 Sending request to:', endpoint);
  console.log('📦 Request payload:', { text: text });
  
  // Send request in format expected by Flask app
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text
    }),
    mode: 'cors'  // Explicitly enable CORS
  });
  
  console.log('📡 Response status:', response.status);
  console.log('📡 Response headers:', [...response.headers.entries()]);
  
  if (!response.ok) {
    const errorText = await response.text();
    console.error('HTTP Error Response:', errorText);
    throw new Error(`HTTP error! status: ${response.status}\nDetails: ${errorText}`);
  }
  
  const data = await response.json();
  console.log('Response data:', data);
  console.log('Response data type:', typeof data);
  console.log('Has analysis property:', 'analysis' in data);
  console.log('Overall confidence:', data.overall_confidence);
  
  // Handle the new LangGraph response format
  if (data.analysis || data.status === 'human_input_required') {
    console.log('✅ Returning full LangGraph response with confidence data');
    return data;
  }
  
  // Fallback for other response formats - but still preserve confidence if available
  console.log('⚠️ Using fallback response format');
  if (typeof data === 'string') {
    return {
      analysis: data,
      overall_confidence: 0.4 // Default low confidence for string responses to trigger refinement UI
    };
  }
  
  // If data has some properties but not analysis, try to preserve confidence
  const fallbackResult = data.response || data.result || data.output || data.message || 'No response received from Bob Ross Helper';
  return {
    analysis: fallbackResult,
    overall_confidence: data.overall_confidence || data.confidence || 0.4, // Default low confidence
    query_type: data.query_type || 'general_help',
    // Preserve any other confidence-related data
    classification_confidence: data.classification_confidence,
    context_confidence: data.context_confidence
  };
}

async function continueWithHumanInput(sessionId, humanFeedback, humanClassification) {
  // Get the LangChain endpoint from storage
  const result = await chrome.storage.sync.get(['langchainEndpoint']);
  let endpoint = result.langchainEndpoint;
  
  // Default to local development endpoint if not configured
  if (!endpoint) {
    endpoint = 'http://127.0.0.1:8080/BobRossHelp';
  }
  
  // Use the continuation endpoint
  const continueEndpoint = endpoint + '/continue';
  
  console.log('🌐 Sending continuation request to:', continueEndpoint);
  console.log('📦 Request payload:', { 
    session_id: sessionId, 
    human_feedback: humanFeedback, 
    human_classification: humanClassification 
  });
  
  // Send request to continuation endpoint
  const response = await fetch(continueEndpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      human_feedback: humanFeedback,
      human_classification: humanClassification
    }),
    mode: 'cors'
  });
  
  console.log('📡 Continuation response status:', response.status);
  
  if (!response.ok) {
    const errorText = await response.text();
    console.error('HTTP Error Response:', errorText);
    throw new Error(`HTTP error! status: ${response.status}\nDetails: ${errorText}`);
  }
  
  const data = await response.json();
  console.log('Continuation response data:', data);
  
  // Handle the continuation response format
  if (data.analysis) {
    console.log('✅ Returning continuation response');
    return data;
  }
  
  // Fallback for other response formats
  return data.response || data.result || data.output || data.message || 'No response received from continuation';
}

chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
  if (message.action === 'textSelected') {
    // Store the selected text for potential use
    chrome.storage.local.set({ lastSelectedText: message.text });
  } else if (message.action === 'rerunWithContext') {
    try {
      console.log('🔄 Rerunning with additional context...');
      const enhancedText = `${message.originalText}\n\nAdditional context: ${message.additionalContext}`;
      const responseData = await queryLangChain(enhancedText);
      
      chrome.runtime.sendMessage({
        target: 'sidepanel',
        action: 'showResult',
        result: responseData.analysis || responseData,
        fullResponse: responseData,
        originalText: message.originalText,
        isRerun: true
      });
      
      sendResponse({ success: true });
    } catch (error) {
      console.error('❌ Error in rerun:', error);
      chrome.runtime.sendMessage({
        target: 'sidepanel',
        action: 'showResult',
        result: `Sorry, there was an error processing your request: ${error.message}`,
        fullResponse: { error: error.message, overall_confidence: 0.0 },
        isRerun: true
      });
      sendResponse({ success: false, error: error.message });
    }
  } else if (message.action === 'continueWithHumanInput') {
    try {
      console.log('🤚 Continuing with human input...');
      console.log('Session ID:', message.sessionId);
      console.log('Human feedback:', message.humanFeedback);
      console.log('Human classification:', message.humanClassification);
      
      const responseData = await continueWithHumanInput(
        message.sessionId,
        message.humanFeedback,
        message.humanClassification
      );
      
      chrome.runtime.sendMessage({
        target: 'sidepanel',
        action: 'showResult',
        result: responseData.analysis || responseData,
        fullResponse: responseData,
        originalText: responseData.original_text || 'Unknown',
        isRerun: true
      });
      
      sendResponse({ success: true });
    } catch (error) {
      console.error('❌ Error in human input continuation:', error);
      chrome.runtime.sendMessage({
        target: 'sidepanel',
        action: 'showResult',
        result: `Sorry, there was an error processing your input: ${error.message}`,
        fullResponse: { error: error.message, overall_confidence: 0.0 },
        isRerun: true
      });
      sendResponse({ success: false, error: error.message });
    }
  }
  return true; // Keep message channel open for async response
});