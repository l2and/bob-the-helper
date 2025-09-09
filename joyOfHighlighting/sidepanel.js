let currentResult = '';
let currentSessionId = null;
let availableClassifications = [];

document.addEventListener('DOMContentLoaded', () => {
    console.log('🎨 Side panel loaded');
    
    // Set up event listeners
    setupEventListeners();
    
    // Listen for messages from background script
    chrome.runtime.onMessage.addListener(handleMessage);
    
    // Show welcome screen initially
    showWelcomeScreen();
});

function setupEventListeners() {
    // Copy button
    const copyBtn = document.getElementById('copyBtn');
    copyBtn.addEventListener('click', copyToClipboard);
    
    // Clear button
    const clearBtn = document.getElementById('clearBtn');
    clearBtn.addEventListener('click', clearResults);
    
    // Rerun button
    const rerunButton = document.getElementById('rerunButton');
    rerunButton.addEventListener('click', handleRerun);
}

function handleMessage(message, sender, sendResponse) {
    console.log('📨 Side panel received message:', message);
    
    if (message.target !== 'sidepanel') {
        return; // Not for us
    }
    
    switch (message.action) {
        case 'showResult':
            showResult(message.result, message.fullResponse, message.originalText, message.isRerun);
            break;
        case 'updateProgress':
            updateProgress(message.message, message.progress);
            break;
        case 'showProgress':
            showProgress();
            break;
        default:
            console.log('Unknown action:', message.action);
    }
    
    sendResponse({ success: true });
}

function showWelcomeScreen() {
    const welcomeScreen = document.getElementById('welcomeScreen');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    const copyBtn = document.getElementById('copyBtn');
    
    welcomeScreen.style.display = 'flex';
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';
    copyBtn.style.display = 'none';
    
    updateFooter('Welcome! Right-click on selected text to get started.');
}

function showProgress() {
    const welcomeScreen = document.getElementById('welcomeScreen');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    
    welcomeScreen.style.display = 'none';
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
    
    // Reset progress
    const progressFill = document.getElementById('progressFill');
    const progressMessage = document.getElementById('progressMessage');
    progressFill.style.width = '0%';
    progressMessage.textContent = 'Starting analysis...';
    
    // Animate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90; // Don't complete until we get results
        progressFill.style.width = progress + '%';
        
        if (progress > 25) progressMessage.textContent = 'Analyzing text...';
        if (progress > 50) progressMessage.textContent = 'Generating response...';
        if (progress > 75) progressMessage.textContent = 'Almost done...';
    }, 200);
    
    // Store interval ID to clear it later
    window.currentProgressInterval = progressInterval;
}

function updateProgress(message, progress) {
    const progressFill = document.getElementById('progressFill');
    const progressMessage = document.getElementById('progressMessage');
    
    if (progress !== undefined) {
        progressFill.style.width = progress + '%';
    }
    
    if (message) {
        progressMessage.textContent = message;
    }
}

function showResult(result, fullResponse, originalText, isRerun = false) {
    console.log('🎭 Showing result in side panel:', result);
    console.log('📊 Full response data:', fullResponse);
    
    // Clear any progress interval
    if (window.currentProgressInterval) {
        clearInterval(window.currentProgressInterval);
        window.currentProgressInterval = null;
    }
    
    // Complete progress bar
    const progressFill = document.getElementById('progressFill');
    const progressMessage = document.getElementById('progressMessage');
    progressFill.style.width = '100%';
    progressMessage.textContent = 'Complete!';
    
    // Show results section after a brief delay
    setTimeout(() => {
        const welcomeScreen = document.getElementById('welcomeScreen');
        const progressSection = document.getElementById('progressSection');
        const resultsSection = document.getElementById('resultsSection');
        const copyBtn = document.getElementById('copyBtn');
        
        welcomeScreen.style.display = 'none';
        progressSection.style.display = 'none';
        resultsSection.style.display = 'flex';
        copyBtn.style.display = 'flex';
        
        // Store current result for copying
        currentResult = result;
        currentSessionId = fullResponse?.session_id || fullResponse?.sessionId;
        availableClassifications = fullResponse?.available_classifications || [];
        
        // Display the result
        displayResult(result, fullResponse, originalText, isRerun);
        
        updateFooter(isRerun ? 'Rerun complete • Click 📋 to copy • Powered by LangGraph + Claude' : 
                     'Click 📋 to copy • Powered by LangGraph + Claude');
    }, 500);
}

function displayResult(result, fullResponse, originalText, isRerun) {
    // Handle confidence indicator
    let confidence = 1.0;
    if (fullResponse && typeof fullResponse === 'object') {
        confidence = fullResponse.overall_confidence || 
                    fullResponse.context_confidence || 
                    fullResponse.classification_confidence ||
                    fullResponse.confidence || 
                    1.0;
    }
    
    if (!fullResponse || typeof fullResponse === 'string') {
        confidence = 0.4;
    }
    
    const confidenceIndicator = document.getElementById('confidenceIndicator');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceDescription = document.getElementById('confidenceDescription');
    
    if (fullResponse && fullResponse.overall_confidence !== undefined) {
        const confidencePercent = Math.round(confidence * 100);
        const confidenceColor = confidence >= 0.8 ? '#28a745' : confidence >= 0.5 ? '#ffc107' : '#dc3545';
        const confidenceClass = confidence >= 0.8 ? 'confidence-high' : confidence >= 0.5 ? 'confidence-medium' : 'confidence-low';
        
        confidenceIndicator.className = `confidence-indicator ${confidenceClass}`;
        confidenceIndicator.style.display = 'flex';
        confidenceIndicator.style.flexDirection = 'column';
        
        confidenceText.innerHTML = `<div style="font-weight: bold; color: ${confidenceColor};">📊 Confidence: ${confidencePercent}%</div>`;
        confidenceDescription.innerHTML = `<div style="font-size: 12px; color: #666;">
            ${confidence >= 0.8 ? 'High confidence response' : 
              confidence >= 0.5 ? 'Moderate confidence - may benefit from more context' : 
              'Low confidence - additional context recommended'}
        </div>`;
    } else {
        confidenceIndicator.style.display = 'none';
    }
    
    // Handle human input required
    const isHumanInputRequired = fullResponse?.status === 'human_input_required';
    const humanInputMessage = document.getElementById('humanInputMessage');
    const humanInputText = document.getElementById('humanInputText');
    const responseContent = document.getElementById('responseContent');
    
    if (isHumanInputRequired) {
        const message = fullResponse?.message || "I need more context to help you better.";
        humanInputMessage.style.display = 'block';
        humanInputText.textContent = message;
        responseContent.innerHTML = ''; // Clear normal response
    } else {
        humanInputMessage.style.display = 'none';
        const content = typeof result === 'string' ? result : fullResponse?.analysis || 'No response available';
        responseContent.innerHTML = content.replace(/\n/g, '<br>');
    }
    
    // Handle refinement UI
    const showRefinementUI = (confidence < 0.70 && !isRerun) || isHumanInputRequired;
    const refinementSection = document.getElementById('refinementSection');
    
    if (showRefinementUI) {
        refinementSection.style.display = 'block';
        setupRefinementUI(originalText, isHumanInputRequired);
    } else {
        refinementSection.style.display = 'none';
    }
}

function setupRefinementUI(originalText, isHumanInputRequired) {
    const buttonContainer = document.getElementById('buttonContainer');
    const contextTextarea = document.getElementById('contextTextarea');
    const rerunButton = document.getElementById('rerunButton');
    
    // Clear existing buttons
    buttonContainer.innerHTML = '';
    
    // Create context buttons
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
        btn.className = 'context-btn';
        btn.onclick = () => handleContextButton(button, originalText, isHumanInputRequired);
        buttonContainer.appendChild(btn);
    });
    
    // Store original text for rerun
    contextTextarea.dataset.originalText = originalText;
}

async function handleContextButton(button, originalText, isHumanInputRequired) {
    // Disable all buttons to prevent double-clicks
    const allButtons = document.querySelectorAll('.context-btn');
    allButtons.forEach(b => b.disabled = true);
    
    // Show loading state on clicked button
    const clickedBtn = event.target;
    const originalText_btn = clickedBtn.textContent;
    clickedBtn.textContent = '🔄 Processing...';
    clickedBtn.style.background = '#ffc107';
    
    try {
        if (isHumanInputRequired && currentSessionId) {
            // Use continuation endpoint for human-in-the-loop with just classification
            console.log(`🚀 Using continueWithHumanInput for session ${currentSessionId}`);
            chrome.runtime.sendMessage({
                action: 'continueWithHumanInput',
                sessionId: currentSessionId,
                humanFeedback: button.context,
                humanClassification: button.value
            });
        } else {
            // Fallback to old rerun method
            console.log('🚀 Using rerunWithContext (no session ID or not human input required)');
            chrome.runtime.sendMessage({
                action: 'rerunWithContext',
                originalText: originalText,
                additionalContext: button.context
            });
        }
        
        // Show progress immediately
        showProgress();
        
    } catch (error) {
        console.error('Failed to continue analysis:', error);
        alert('Failed to continue analysis. Please try again.');
        
        // Reset button state on error
        clickedBtn.textContent = originalText_btn;
        clickedBtn.style.background = '#fff';
        allButtons.forEach(b => b.disabled = false);
    }
}

async function handleRerun() {
    const contextTextarea = document.getElementById('contextTextarea');
    const rerunButton = document.getElementById('rerunButton');
    const originalText = contextTextarea.dataset.originalText || '';
    
    const additionalContext = contextTextarea.value.trim();
    const selectedClassification = contextTextarea.dataset.selectedClassification || '';
    
    if (!additionalContext && !selectedClassification) {
        alert('Please provide some additional context first!');
        contextTextarea.focus();
        return;
    }
    
    // Show loading state
    rerunButton.disabled = true;
    rerunButton.textContent = '🔄 Processing...';
    
    try {
        const isHumanInputRequired = currentSessionId !== null;
        
        if (isHumanInputRequired && currentSessionId) {
            // Use continuation endpoint for human-in-the-loop
            console.log(`🚀 Using continueWithHumanInput for session ${currentSessionId} with custom text`);
            chrome.runtime.sendMessage({
                action: 'continueWithHumanInput',
                sessionId: currentSessionId,
                humanFeedback: additionalContext,
                humanClassification: selectedClassification
            });
        } else {
            // Fallback to old rerun method
            console.log('🚀 Using rerunWithContext with custom text (no session ID or not human input required)');
            chrome.runtime.sendMessage({
                action: 'rerunWithContext',
                originalText: originalText,
                additionalContext: additionalContext
            });
        }
        
        // Show progress immediately
        showProgress();
        
    } catch (error) {
        console.error('Failed to continue analysis:', error);
        alert('Failed to continue analysis. Please try again.');
        rerunButton.disabled = false;
        rerunButton.textContent = '🔄 Get Better Answer';
    }
}

async function copyToClipboard() {
    const copyBtn = document.getElementById('copyBtn');
    
    if (!currentResult) {
        return;
    }
    
    try {
        await navigator.clipboard.writeText(currentResult);
        
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
        textArea.value = currentResult;
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
}

function clearResults() {
    currentResult = '';
    currentSessionId = null;
    availableClassifications = [];
    
    // Clear progress interval if running
    if (window.currentProgressInterval) {
        clearInterval(window.currentProgressInterval);
        window.currentProgressInterval = null;
    }
    
    // Reset form
    const contextTextarea = document.getElementById('contextTextarea');
    const rerunButton = document.getElementById('rerunButton');
    
    contextTextarea.value = '';
    rerunButton.disabled = false;
    rerunButton.textContent = '🔄 Get Better Answer';
    
    // Show welcome screen
    showWelcomeScreen();
}

function updateFooter(message) {
    const footer = document.getElementById('footer');
    footer.textContent = message;
}