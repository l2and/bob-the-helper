let selectedText = '';

document.addEventListener('mouseup', () => {
  const selection = window.getSelection();
  if (selection.rangeCount > 0) {
    selectedText = selection.toString().trim();
    if (selectedText.length > 0) {
      chrome.runtime.sendMessage({
        action: 'textSelected',
        text: selectedText
      });
    }
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getSelectedText') {
    sendResponse({ text: selectedText });
  } else if (message.action === 'showResult') {
    showResultModal(message.result);
  }
});

function showResultModal(result) {
  const existingModal = document.getElementById('bob-ross-modal');
  if (existingModal) {
    existingModal.remove();
  }

  const modal = document.createElement('div');
  modal.id = 'bob-ross-modal';
  modal.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    width: 400px;
    max-height: 500px;
    background: white;
    border: 2px solid #8B4513;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    z-index: 10000;
    font-family: Arial, sans-serif;
    overflow-y: auto;
  `;

  const header = document.createElement('div');
  header.style.cssText = `
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 10px;
  `;

  const title = document.createElement('h3');
  title.textContent = '🎨 Bob Ross Helper';
  title.style.cssText = 'margin: 0; color: #8B4513;';

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '×';
  closeBtn.style.cssText = `
    background: none;
    border: none;
    font-size: 24px;
    cursor: pointer;
    color: #666;
  `;
  closeBtn.onclick = () => modal.remove();

  const content = document.createElement('div');
  content.innerHTML = result.replace(/\n/g, '<br>');
  content.style.cssText = 'line-height: 1.5; color: #333;';

  header.appendChild(title);
  header.appendChild(closeBtn);
  modal.appendChild(header);
  modal.appendChild(content);
  document.body.appendChild(modal);

  setTimeout(() => modal.remove(), 30000);
}