let selectedText = '';

document.addEventListener('mouseup', () => {
  const selection = window.getSelection();
  if (selection.rangeCount > 0) {
    selectedText = selection.toString().trim();
    if (selectedText.length > 0) {
      try {
        chrome.runtime.sendMessage({
          action: 'textSelected',
          text: selectedText
        });
      } catch (error) {
        if (error.message.includes('Extension context invalidated')) {
          console.log('🔄 Extension reloaded. Please refresh this page to continue using Bob Ross Helper.');
        } else {
          console.error('Error sending message:', error);
        }
      }
    }
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('📨 Content script received message:', message);
  
  if (message.action === 'getSelectedText') {
    console.log('📤 Sending selected text:', selectedText);
    sendResponse({ text: selectedText });
  } else if (message.action === 'showResult') {
    console.log('🎨 Showing result modal with:', message.result);
    try {
      showResultModal(message.result, message.fullResponse, message.originalText, message.isRerun);
      console.log('✅ Modal shown successfully');
      sendResponse({ success: true });
    } catch (error) {
      console.error('❌ Error showing modal:', error);
      sendResponse({ success: false, error: error.message });
    }
  }
  
  return true; // Keep message channel open for async response
});

function showResultModal(result, fullResponse = null, originalText = '', isRerun = false) {
  console.log('🎭 showResultModal called with result:', result);
  console.log('📊 Full response data:', fullResponse);
  
  // Clean up any existing modal
  const existingModal = document.getElementById('bob-ross-modal');
  const existingOverlay = document.getElementById('bob-ross-overlay');
  if (existingModal) {
    console.log('🧹 Removing existing modal');
    existingModal.remove();
  }
  if (existingOverlay) {
    console.log('🧹 Removing existing overlay');
    existingOverlay.remove();
  }

  // Create overlay to prevent clicking outside
  const overlay = document.createElement('div');
  overlay.id = 'bob-ross-overlay';
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 9999;
  `;

  const modal = document.createElement('div');
  modal.id = 'bob-ross-modal';
  modal.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 500px;
    max-width: 90vw;
    max-height: 70vh;
    background: white;
    border: 3px solid #8B4513;
    border-radius: 12px;
    padding: 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    z-index: 10000;
    font-family: Georgia, serif;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  `;

  const header = document.createElement('div');
  header.style.cssText = `
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    background: linear-gradient(135deg, #8B4513 0%, #A0522D 100%);
    color: white;
    border-bottom: none;
  `;

  const title = document.createElement('h3');
  title.textContent = '🎨 Bob Ross Helper';
  title.style.cssText = 'margin: 0; color: white; font-size: 18px; font-weight: bold;';

  // Header buttons container
  const buttonsContainer = document.createElement('div');
  buttonsContainer.style.cssText = 'display: flex; gap: 8px; align-items: center;';

  // Copy button
  const copyBtn = document.createElement('button');
  copyBtn.textContent = '📋';
  copyBtn.title = 'Copy to clipboard';
  copyBtn.style.cssText = `
    background: rgba(255, 255, 255, 0.2);
    border: 2px solid white;
    border-radius: 6px;
    width: 32px;
    height: 32px;
    font-size: 16px;
    cursor: pointer;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  `;
  copyBtn.onmouseover = () => {
    copyBtn.style.background = 'rgba(255, 255, 255, 0.3)';
    copyBtn.style.transform = 'scale(1.1)';
  };
  copyBtn.onmouseout = () => {
    copyBtn.style.background = 'rgba(255, 255, 255, 0.2)';
    copyBtn.style.transform = 'scale(1)';
  };
  copyBtn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(result);
      // Visual feedback
      const originalText = copyBtn.textContent;
      copyBtn.textContent = '✓';
      copyBtn.style.background = 'rgba(0, 255, 0, 0.3)';
      setTimeout(() => {
        copyBtn.textContent = originalText;
        copyBtn.style.background = 'rgba(255, 255, 255, 0.2)';
      }, 1000);
      console.log('✅ Text copied to clipboard');
    } catch (err) {
      console.error('❌ Failed to copy text:', err);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = result;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      
      // Visual feedback
      const originalText = copyBtn.textContent;
      copyBtn.textContent = '✓';
      copyBtn.style.background = 'rgba(0, 255, 0, 0.3)';
      setTimeout(() => {
        copyBtn.textContent = originalText;
        copyBtn.style.background = 'rgba(255, 255, 255, 0.2)';
      }, 1000);
      console.log('✅ Text copied to clipboard (fallback method)');
    }
  };

  // Close button
  const closeBtn = document.createElement('button');
  closeBtn.textContent = '×';
  closeBtn.title = 'Close';
  closeBtn.style.cssText = `
    background: rgba(255, 255, 255, 0.2);
    border: 2px solid white;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    font-size: 20px;
    cursor: pointer;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  `;
  closeBtn.onmouseover = () => {
    closeBtn.style.background = 'rgba(255, 255, 255, 0.3)';
    closeBtn.style.transform = 'scale(1.1)';
  };
  closeBtn.onmouseout = () => {
    closeBtn.style.background = 'rgba(255, 255, 255, 0.2)';
    closeBtn.style.transform = 'scale(1)';
  };
  closeBtn.onclick = () => {
    overlay.remove();
    modal.remove();
  };

  // Check if this is a human-in-the-loop response that needs user input
  console.log('📊 Full response object:', fullResponse);
  console.log('📊 Full response type:', typeof fullResponse);
  
  const isHumanInputRequired = fullResponse?.status === 'human_input_required';
  const sessionId = fullResponse?.session_id || fullResponse?.sessionId;
  const availableClassifications = fullResponse?.available_classifications || [];
  
  console.log(`🔍 Debug info: isHumanInputRequired=${isHumanInputRequired}, sessionId=${sessionId}, status=${fullResponse?.status}`);
  
  let confidence = 1.0; // Default high confidence
  
  if (fullResponse && typeof fullResponse === 'object') {
    // Try different confidence properties
    confidence = fullResponse.overall_confidence || 
                fullResponse.context_confidence || 
                fullResponse.classification_confidence ||
                fullResponse.confidence || 
                1.0;
  }
  
  // If no fullResponse or it's a string, assume low confidence to show refinement UI
  if (!fullResponse || typeof fullResponse === 'string') {
    confidence = 0.4;
  }
  
  const showRefinementUI = (confidence < 0.70 && !isRerun) || isHumanInputRequired;

  console.log(`📊 Confidence level: ${confidence}, Show refinement UI: ${showRefinementUI}, Human input required: ${isHumanInputRequired}`);

  const content = document.createElement('div');
  content.style.cssText = `
    padding: 25px;
    line-height: 1.6; 
    color: #333;
    overflow-y: auto;
    flex: 1;
    font-size: 15px;
  `;

  // Add confidence indicator if available
  if (fullResponse && fullResponse.overall_confidence !== undefined) {
    const confidenceIndicator = document.createElement('div');
    const confidencePercent = Math.round(confidence * 100);
    const confidenceColor = confidence >= 0.8 ? '#28a745' : confidence >= 0.5 ? '#ffc107' : '#dc3545';
    
    confidenceIndicator.style.cssText = `
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 15px;
      padding: 10px;
      background: #f8f9fa;
      border-radius: 6px;
      border-left: 4px solid ${confidenceColor};
    `;
    
    confidenceIndicator.innerHTML = `
      <div style="font-weight: bold; color: ${confidenceColor};">
        📊 Confidence: ${confidencePercent}%
      </div>
      <div style="font-size: 12px; color: #666;">
        ${confidence >= 0.8 ? 'High confidence response' : 
          confidence >= 0.5 ? 'Moderate confidence - may benefit from more context' : 
          'Low confidence - additional context recommended'}
      </div>
    `;
    
    content.appendChild(confidenceIndicator);
  }

  // Add main response content
  const responseContent = document.createElement('div');
  
  if (isHumanInputRequired) {
    // Show the message asking for human input instead of analysis
    const message = fullResponse?.message || "I need more context to help you better.";
    responseContent.innerHTML = `
      <div style="background: #e3f2fd; padding: 15px; border-radius: 6px; border-left: 4px solid #2196f3; margin-bottom: 15px;">
        <div style="font-weight: bold; color: #1976d2; margin-bottom: 8px;">🤚 I need your help!</div>
        <div style="color: #1565c0;">${message}</div>
      </div>
    `;
  } else {
    // Show normal analysis response
    responseContent.innerHTML = (typeof result === 'string' ? result : fullResponse?.analysis || 'No response available').replace(/\n/g, '<br>');
  }
  
  content.appendChild(responseContent);

  // Add context refinement UI for low confidence responses
  if (showRefinementUI) {
    const refinementSection = document.createElement('div');
    refinementSection.style.cssText = `
      margin-top: 20px;
      padding: 15px;
      background: #fff3cd;
      border: 1px solid #ffeaa7;
      border-radius: 6px;
    `;

    const refinementTitle = document.createElement('h4');
    refinementTitle.textContent = '🎯 Need more specific help?';
    refinementTitle.style.cssText = 'margin: 0 0 10px 0; color: #856404;';

    const refinementDescription = document.createElement('p');
    refinementDescription.innerHTML = 'I\'m not entirely confident about this response. You can help me provide a better answer by:<br><strong>Click a button below to continue immediately, or provide your own context:</strong>';
    refinementDescription.style.cssText = 'margin: 0 0 15px 0; font-size: 13px; color: #856404; line-height: 1.4;';

    // Context buttons - use server-provided classifications if available
    const buttonContainer = document.createElement('div');
    buttonContainer.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px;';

    const contextButtons = availableClassifications.length > 0 ? 
      availableClassifications.map(cls => ({
        text: cls.label,
        context: cls.description,
        value: cls.value
      })) :
      [
        { text: '🐛 This is an error', context: 'I am seeing an error or exception message', value: 'error_help' },
        { text: '📖 Explain concept', context: 'I want to understand what this concept or term means', value: 'concept_learning' },
        { text: '⚙️ Show usage', context: 'I want to know how to use this API or function', value: 'api_usage' },
        { text: '🔍 Code review', context: 'I want to understand what this code does', value: 'code_explanation' },
        { text: '🚀 Implementation help', context: 'I want help implementing or building something', value: 'implementation_help' }
      ];

    contextButtons.forEach(button => {
      const btn = document.createElement('button');
      btn.textContent = button.text;
      btn.style.cssText = `
        background: #fff;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
      `;
      btn.onmouseover = () => btn.style.background = '#ffc107';
      btn.onmouseout = () => btn.style.background = '#fff';
      btn.onclick = async () => {
        // Disable all buttons to prevent double-clicks
        const allButtons = refinementSection.querySelectorAll('button');
        allButtons.forEach(b => b.disabled = true);
        
        // Show loading state on clicked button
        const originalText = btn.textContent;
        btn.textContent = '🔄 Processing...';
        btn.style.background = '#ffc107';
        
        try {
          if (isHumanInputRequired && sessionId) {
            // Use continuation endpoint for human-in-the-loop with just classification
            console.log(`🚀 Using continueWithHumanInput for session ${sessionId}`);
            await chrome.runtime.sendMessage({
              action: 'continueWithHumanInput',
              sessionId: sessionId,
              humanFeedback: button.context, // Use the button's description as context
              humanClassification: button.value
            });
          } else {
            // Fallback to old rerun method
            console.log('🚀 Using rerunWithContext (no session ID or not human input required)');
            await chrome.runtime.sendMessage({
              action: 'rerunWithContext',
              originalText: originalText,
              additionalContext: button.context
            });
          }
          
          // Close the current modal immediately after sending the request
          // The new response will create a fresh modal
          console.log('🚀 Request sent successfully, closing current modal to wait for new response');
          overlay.remove();
          modal.remove();
          
        } catch (error) {
          console.error('Failed to continue analysis:', error);
          alert('Failed to continue analysis. Please try again.');
          // Reset button state on error
          btn.textContent = originalText;
          btn.style.background = '#fff';
          allButtons.forEach(b => b.disabled = false);
        }
      };
      buttonContainer.appendChild(btn);
    });

    // Custom text input
    const textAreaLabel = document.createElement('label');
    textAreaLabel.textContent = 'Or provide your own context:';
    textAreaLabel.style.cssText = 'display: block; margin-bottom: 8px; font-weight: bold; font-size: 13px; color: #856404;';

    const textArea = document.createElement('textarea');
    textArea.placeholder = 'Add more context about what you\'re trying to do or understand...';
    textArea.style.cssText = `
      width: 100%;
      height: 60px;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 4px;
      font-size: 13px;
      resize: vertical;
      box-sizing: border-box;
      font-family: Arial, sans-serif;
      background-color: white;
      color: black;
    `;

    // Rerun button
    const rerunButton = document.createElement('button');
    rerunButton.textContent = '🔄 Get Better Answer';
    rerunButton.style.cssText = `
      background: #28a745;
      color: white;
      border: none;
      border-radius: 4px;
      padding: 10px 20px;
      font-size: 13px;
      cursor: pointer;
      margin-top: 10px;
      transition: background 0.2s;
    `;
    rerunButton.onmouseover = () => rerunButton.style.background = '#218838';
    rerunButton.onmouseout = () => rerunButton.style.background = '#28a745';
    
    rerunButton.onclick = async () => {
      const additionalContext = textArea.value.trim();
      const selectedClassification = textArea.dataset.selectedClassification || '';
      
      if (!additionalContext && !selectedClassification) {
        alert('Please provide some additional context or select a classification first!');
        textArea.focus();
        return;
      }

      // Show loading state
      rerunButton.disabled = true;
      rerunButton.textContent = '🔄 Processing...';
      
      try {
        if (isHumanInputRequired && sessionId) {
          // Use continuation endpoint for human-in-the-loop
          console.log(`🚀 Using continueWithHumanInput for session ${sessionId} with custom text`);
          await chrome.runtime.sendMessage({
            action: 'continueWithHumanInput',
            sessionId: sessionId,
            humanFeedback: additionalContext,
            humanClassification: selectedClassification
          });
        } else {
          // Fallback to old rerun method
          console.log('🚀 Using rerunWithContext with custom text (no session ID or not human input required)');
          await chrome.runtime.sendMessage({
            action: 'rerunWithContext',
            originalText: originalText,
            additionalContext: additionalContext
          });
        }
        
        // Close the current modal immediately after sending the request
        // The new response will create a fresh modal
        console.log('🚀 Custom context request sent successfully, closing current modal to wait for new response');
        overlay.remove();
        modal.remove();
        
      } catch (error) {
        console.error('Failed to continue analysis:', error);
        alert('Failed to continue analysis. Please try again.');
        rerunButton.disabled = false;
        rerunButton.textContent = '🔄 Get Better Answer';
      }
    };

    refinementSection.appendChild(refinementTitle);
    refinementSection.appendChild(refinementDescription);
    refinementSection.appendChild(buttonContainer);
    refinementSection.appendChild(textAreaLabel);
    refinementSection.appendChild(textArea);
    refinementSection.appendChild(rerunButton);
    
    content.appendChild(refinementSection);
  }

  // Add footer with helpful info
  const footer = document.createElement('div');
  footer.style.cssText = `
    padding: 15px 25px;
    background: #f8f9fa;
    border-top: 1px solid #ddd;
    font-size: 12px;
    color: #666;
    text-align: center;
  `;
  footer.textContent = isRerun ? 
    'Rerun complete • Click 📋 to copy • Click × to close • Powered by LangGraph + Claude' :
    'Click 📋 to copy • Click × to close • Powered by LangGraph + Claude';

  buttonsContainer.appendChild(copyBtn);
  buttonsContainer.appendChild(closeBtn);
  
  header.appendChild(title);
  header.appendChild(buttonsContainer);
  modal.appendChild(header);
  modal.appendChild(content);
  modal.appendChild(footer);
  
  // Prevent clicking outside to close
  overlay.onclick = (e) => {
    if (e.target === overlay) {
      // Don't close - require X button
      modal.style.animation = 'shake 0.5s ease-in-out';
      setTimeout(() => {
        modal.style.animation = '';
      }, 500);
    }
  };

  // Add shake animation
  const style = document.createElement('style');
  style.textContent = `
    @keyframes shake {
      0%, 100% { transform: translate(-50%, -50%); }
      10%, 30%, 50%, 70%, 90% { transform: translate(-52%, -50%); }
      20%, 40%, 60%, 80% { transform: translate(-48%, -50%); }
    }
  `;
  document.head.appendChild(style);

  console.log('📱 Adding overlay and modal to document body');
  document.body.appendChild(overlay);
  document.body.appendChild(modal);

  // Focus the close button for accessibility
  closeBtn.focus();
  
  console.log('🎉 Modal creation completed successfully');
}