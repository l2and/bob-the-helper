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
    console.log('🎨 Showing result modal with:', message.result.substring(0, 100) + '...');
    try {
      showResultModal(message.result);
      console.log('✅ Modal shown successfully');
      sendResponse({ success: true });
    } catch (error) {
      console.error('❌ Error showing modal:', error);
      sendResponse({ success: false, error: error.message });
    }
  }
  
  return true; // Keep message channel open for async response
});

function showResultModal(result) {
  console.log('🎭 showResultModal called with result length:', result.length);
  
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

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '×';
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

  const content = document.createElement('div');
  content.innerHTML = result.replace(/\n/g, '<br>');
  content.style.cssText = `
    padding: 25px;
    line-height: 1.6; 
    color: #333;
    overflow-y: auto;
    flex: 1;
    font-size: 15px;
  `;

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
  footer.textContent = 'Click the × button to close • Powered by LangGraph + Claude';

  header.appendChild(title);
  header.appendChild(closeBtn);
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