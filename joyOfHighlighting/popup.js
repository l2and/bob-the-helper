document.addEventListener('DOMContentLoaded', () => {
    const endpointInput = document.getElementById('endpoint');
    const saveButton = document.getElementById('save');
    const testButton = document.getElementById('test');
    const statusDiv = document.getElementById('status');

    // Load saved endpoint on startup
    loadSavedEndpoint();

    // Event listeners
    saveButton.addEventListener('click', saveConfiguration);
    testButton.addEventListener('click', testConnection);

    async function loadSavedEndpoint() {
        try {
            const result = await chrome.storage.sync.get(['langchainEndpoint']);
            if (result.langchainEndpoint) {
                endpointInput.value = result.langchainEndpoint;
            }
        } catch (error) {
            console.error('Error loading saved endpoint:', error);
        }
    }

    async function saveConfiguration() {
        const endpoint = endpointInput.value.trim();
        
        if (!endpoint) {
            showStatus('Please enter an endpoint URL', 'error');
            return;
        }

        // Basic URL validation
        try {
            new URL(endpoint);
        } catch (error) {
            showStatus('Please enter a valid URL', 'error');
            return;
        }

        try {
            await chrome.storage.sync.set({ langchainEndpoint: endpoint });
            showStatus('Configuration saved successfully!', 'success');
            console.log('✅ Configuration saved:', endpoint);
        } catch (error) {
            console.error('❌ Error saving configuration:', error);
            showStatus('Error saving configuration: ' + error.message, 'error');
        }
    }

    async function testConnection() {
        const endpoint = endpointInput.value.trim();
        
        if (!endpoint) {
            showStatus('Please enter an endpoint URL first', 'error');
            return;
        }

        // Basic URL validation
        try {
            new URL(endpoint);
        } catch (error) {
            showStatus('Please enter a valid URL', 'error');
            return;
        }

        // Disable test button and show loading
        testButton.disabled = true;
        testButton.textContent = 'Testing...';
        showStatus('Testing connection...', 'success');

        try {
            console.log('🌐 Testing connection to:', endpoint);
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: 'Test connection from Bob Ross Helper extension'
                }),
                mode: 'cors'
            });

            console.log('📡 Test response status:', response.status);

            if (response.ok) {
                const data = await response.json();
                console.log('✅ Test response data:', data);
                showStatus('✅ Connection successful! Server is responding.', 'success');
            } else {
                const errorText = await response.text();
                console.error('❌ Test failed with status:', response.status);
                showStatus(`❌ Connection failed: HTTP ${response.status}\nDetails: ${errorText}`, 'error');
            }
        } catch (error) {
            console.error('❌ Connection test error:', error);
            let errorMessage = '❌ Connection failed: ';
            
            if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
                errorMessage += 'Unable to reach the server. Please check:\n• Server is running\n• URL is correct\n• CORS is enabled on server';
            } else {
                errorMessage += error.message;
            }
            
            showStatus(errorMessage, 'error');
        } finally {
            // Re-enable test button
            testButton.disabled = false;
            testButton.textContent = 'Test Connection';
        }
    }

    function showStatus(message, type) {
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
        statusDiv.style.display = 'block';
        
        // Auto-hide success messages after 3 seconds
        if (type === 'success') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
    }
});