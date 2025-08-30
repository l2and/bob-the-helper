document.addEventListener('DOMContentLoaded', async () => {
    const endpointInput = document.getElementById('endpoint');
    const saveButton = document.getElementById('save');
    const testButton = document.getElementById('test');
    const statusDiv = document.getElementById('status');

    // Load saved endpoint or set default
    const result = await chrome.storage.sync.get(['langchainEndpoint']);
    if (result.langchainEndpoint) {
        endpointInput.value = result.langchainEndpoint;
    } else {
        // Set default endpoint for local development
        endpointInput.value = 'http://127.0.0.1:8080/BobRossHelp';
    }

    saveButton.addEventListener('click', async () => {
        const endpoint = endpointInput.value.trim();
        
        if (!endpoint) {
            showStatus('Please enter a valid endpoint URL', 'error');
            return;
        }

        try {
            await chrome.storage.sync.set({ langchainEndpoint: endpoint });
            showStatus('Configuration saved successfully!', 'success');
        } catch (error) {
            showStatus('Error saving configuration: ' + error.message, 'error');
        }
    });

    testButton.addEventListener('click', async () => {
        const endpoint = endpointInput.value.trim();
        
        if (!endpoint) {
            showStatus('Please enter an endpoint URL first', 'error');
            return;
        }

        showStatus('Testing connection...', 'success');
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: 'test connection'
                })
            });

            if (response.ok) {
                const data = await response.json();
                showStatus('✅ Connection successful! Response received.', 'success');
            } else {
                showStatus(`❌ Connection failed: HTTP ${response.status}`, 'error');
            }
        } catch (error) {
            showStatus('❌ Connection failed: ' + error.message, 'error');
        }
    });

    function showStatus(message, type) {
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
        statusDiv.style.display = 'block';
        
        if (type === 'success') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
    }
});