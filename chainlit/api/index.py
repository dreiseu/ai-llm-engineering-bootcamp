import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Create FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chainlit Echo Bot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin: 20px 0; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background-color: #e3f2fd; text-align: right; }
            .bot { background-color: #f5f5f5; }
            .input-container { display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; }
            button { padding: 10px 20px; }
        </style>
    </head>
    <body>
        <h1>Chainlit Echo Bot</h1>
        <div class="chat-container" id="chat"></div>
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type your message...">
            <button onclick="sendMessage()">Send</button>
        </div>
        
        <script>
            function addMessage(content, isUser) {
                const chat = document.getElementById('chat');
                const message = document.createElement('div');
                message.className = 'message ' + (isUser ? 'user' : 'bot');
                message.textContent = content;
                chat.appendChild(message);
                chat.scrollTop = chat.scrollHeight;
            }
            
            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (message) {
                    addMessage(message, true);
                    addMessage('Echo: ' + message, false);
                    input.value = '';
                }
            }
            
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Initial greeting
            addMessage('Hello! I\\'m a simple echo bot. Send me a message and I\\'ll echo it back to you!', false);
        </script>
    </body>
    </html>
    """)

@app.post("/chat")
async def chat(message: dict):
    user_message = message.get("content", "")
    response = f"Echo: {user_message}"
    return {"content": response}