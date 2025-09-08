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
            // Use the simple /test endpoint instead of triggering full workflow
            let testEndpoint = endpoint;
            if (endpoint.endsWith('/BobRossHelp')) {
                testEndpoint = endpoint.replace('/BobRossHelp', '/test');
            } else if (!endpoint.endsWith('/test')) {
                testEndpoint = endpoint.replace(/\/$/, '') + '/test';
            }

            const response = await fetch(testEndpoint, {
                method: 'GET'
            });

            if (response.ok) {
                const data = await response.json();
                showStatus(`✅ ${data.message || 'Connection successful!'}`, 'success');
                console.log('Test response:', data);
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