document.addEventListener('DOMContentLoaded', () => {
    const characterSelection = document.getElementById('character-selection');
    const chatContainer = document.getElementById('chat-container');
    const startChatBtn = document.getElementById('start-chat-btn');
    const characterDropdown = document.getElementById('character-dropdown');
    const characterNameSpan = document.getElementById('character-name');
    const chatBox = document.getElementById('chat-box');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const emotionSelect = document.getElementById('emotion-select');

    // Handle starting the chat
    startChatBtn.addEventListener('click', async () => {
        const selectedCharacter = characterDropdown.value;

        const response = await fetch('/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ character: selectedCharacter })
        });

        const data = await response.json();

        if (data.status === 'success') {
            characterNameSpan.textContent = data.character;
            characterSelection.classList.add('hidden');
            chatContainer.classList.remove('hidden');
            addMessage('agent', `Hello! You are now playing as a ${data.character}. What would you like to talk about?`);
        }
    });

    // Handle sending a message
    messageForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        const emotion = emotionSelect.value;

        if (!message) return;

        addMessage('user', message, emotion);
        messageInput.value = '';

        // Show loading speech bubble
        const loadingElement = addLoadingMessage();

        const response = await fetch('/send', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, emotion })
        });

        const data = await response.json();
        
        // Remove loading message
        if (loadingElement) {
            loadingElement.remove();
        }
        
        if (data.agent_message) {
            addMessage('agent', data.agent_message, null, data.detected_emotion, data.target_emotion);
        }
    });

    function addMessage(role, message, selfReportedEmotion = null, detectedEmotion = null, targetEmotion = null) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', role === 'user' ? 'user-message' : 'agent-message');
        
        let innerHTML = `<span>${message}</span>`;
        
        if (role === 'user') {
            if (selfReportedEmotion) {
                innerHTML += `<div class="emotion-tag self-reported">You felt: ${selfReportedEmotion}</div>`;
            }
            if (detectedEmotion) {
                innerHTML += `<div class="emotion-tag detected">AI detected: ${detectedEmotion}</div>`;
            }
        } else if (role === 'agent' && targetEmotion) {
            innerHTML += `<div class="emotion-tag target">Targeting: ${targetEmotion}</div>`;
        }
        
        messageElement.innerHTML = innerHTML;

        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function addLoadingMessage() {
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('message', 'agent-message', 'loading-message');
        loadingElement.innerHTML = '<span class="loading-text">. . .</span>';
        
        chatBox.appendChild(loadingElement);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        return loadingElement;
    }
}); 