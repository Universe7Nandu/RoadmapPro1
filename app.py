# app.py
import streamlit as st
import re
import os
import time
import random
from dotenv import load_dotenv
from groq import Groq

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="AI Assistant | Nandesh Kalashetti",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Configure Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define CSS styles for modern UI
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Base styling */
    body {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Chat interface */
    .chat-interface {
        height: 100vh;
        display: flex;
        flex-direction: column;
        padding: 0;
        background-color: #0f172a;
        overflow: hidden;
        position: relative;
    }
    
    /* Chat title */
    .chat-title {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 10px;
        margin-top: 0;
        position: absolute;
        top: 0;
        width: 100%;
        z-index: 50;
        background: linear-gradient(180deg, rgba(15, 23, 42, 1) 0%, rgba(15, 23, 42, 0.9) 80%, rgba(15, 23, 42, 0) 100%);
        pointer-events: none;
    }
    
    .chat-title h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    
    /* Messages container */
    .messages-container {
        flex: 1;
        overflow-y: auto;
        padding: 70px 20px 100px 20px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        scroll-behavior: smooth;
        max-width: 1000px;
        margin: 0 auto;
        width: 100%;
    }
    
    /* Message styles */
    .message {
        padding: 12px 16px;
        margin-bottom: 8px;
        border-radius: 8px;
        max-width: 85%;
        animation: fadeIn 0.3s ease-out;
        word-wrap: break-word;
        line-height: 1.5;
    }
    
    .user-message {
        background-color: #2563eb;
        color: white;
        align-self: flex-end;
        border-radius: 18px 18px 0 18px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .bot-message {
        background-color: #334155;
        color: #f1f5f9;
        align-self: flex-start;
        border-radius: 18px 18px 18px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Input area */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 15px;
        background: linear-gradient(to top, #0f172a 80%, rgba(15, 23, 42, 0));
        z-index: 100;
    }
    
    .input-container {
        display: flex;
        align-items: center;
        background-color: #1e293b;
        border-radius: 50px;
        padding: 5px 5px 5px 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        margin: 0 auto;
        width: 100%;
        max-width: 800px;
    }
    
    .stTextInput input {
        background-color: transparent !important;
        color: #f8fafc !important;
        border: none !important;
        padding: 12px 0 !important;
        font-size: 16px !important;
        width: 100% !important;
    }
    
    /* Send button */
    .send-button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .send-button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Welcome container */
    .welcome-container {
        text-align: center;
        padding: 40px 20px;
        color: #94a3b8;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        margin-top: 40px;
    }
    
    /* Sidebar styling */
    .sidebar .stMarkdown {
        color: #f8fafc !important;
    }
    
    .sidebar-content {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    /* Chat history in sidebar */
    .history-item {
        background-color: #334155;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 8px 0;
        font-size: 13px;
        color: #e2e8f0;
        cursor: pointer;
        transition: all 0.2s;
        border-left: 3px solid transparent;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .history-item:hover {
        background-color: #3b4a63;
        border-left-color: #3b82f6;
    }
    
    .history-container {
        max-height: 300px;
        overflow-y: auto;
        padding-right: 5px;
        margin-top: 10px;
    }
    
    /* Feature cards */
    .feature-card {
        background-color: #334155;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {display:none;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .logo-container img {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #3b82f6;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Tech badges */
    .tech-badge {
        display: inline-block;
        background-color: #334155;
        color: #94a3b8;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 0 5px 5px 0;
    }
    
    /* Connect links */
    .connect-links {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 15px 0;
    }
    
    .connect-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background-color: #334155;
        border-radius: 50%;
        transition: all 0.3s ease;
    }
    
    .connect-link:hover {
        transform: translateY(-3px);
        background-color: #3b82f6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 15px;
        color: #64748b;
        font-size: 12px;
        position: fixed;
        bottom: 0;
        width: 100%;
        z-index: 1;
        background: linear-gradient(to top, rgba(15, 23, 42, 0.8) 0%, rgba(15, 23, 42, 0) 100%);
        pointer-events: none;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .message {
            max-width: 90%;
            font-size: 14px;
        }
        
        .input-container {
            padding: 3px 3px 3px 15px;
        }
        
        .stTextInput input {
            font-size: 14px !important;
        }
        
        .send-button {
            width: 36px;
            height: 36px;
        }
        
        .welcome-container {
            padding: 20px 10px;
        }
        
        .welcome-container img {
            width: 60px;
        }
        
        .welcome-container h3 {
            font-size: 20px !important;
        }
        
        .welcome-container p {
            font-size: 14px !important;
        }
        
        .sidebar-content {
            padding: 15px;
        }
        
        .logo-container img {
            width: 60px;
            height: 60px;
        }
        
        .history-container {
            max-height: 200px;
        }
        
        .chat-title h1 {
            font-size: 22px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'rules' not in st.session_state:
        # Define rule patterns
        st.session_state.rules = [
            {
                'patterns': [r'hello|hi|hey|greetings', r'^hi$'],
                'responses': ["Hello! How can I help you today?", "Hi there! What can I assist you with?"]
            },
            {
                'patterns': [r'who are you|what are you|tell me about yourself'],
                'responses': ["I'm an AI assistant with rule-based capabilities. I can answer your questions based on predefined patterns and use AI for more complex queries."]
            },
            {
                'patterns': [r'bye|goodbye|see you|farewell'],
                'responses': ["Goodbye! Have a great day!", "See you later! Take care!"]
            },
            {
                'patterns': [r'thank you|thanks'],
                'responses': ["You're welcome!", "Happy to help!"]
            },
            {
                'patterns': [r'what can you do|help|capabilities'],
                'responses': ["I can answer simple questions, provide information, use AI for complex queries, and help with various tasks."]
            },
            {
                'patterns': [r'weather|temperature'],
                'responses': ["I'm sorry, I don't have access to real-time weather data."]
            },
            {
                'patterns': [r'your name'],
                'responses': ["I'm your AI assistant, created by Nandesh Kalashetti."]
            },
            {
                'patterns': [r'how are you'],
                'responses': ["I'm doing well, thank you! How about you?", "I'm functioning properly, thanks for asking!"]
            },
            {
                'patterns': [r'who (is|made) (you|this)|creator|developer'],
                'responses': ["I was created by Nandesh Kalashetti, a full-stack developer specializing in MERN stack, React.js, TypeScript, PHP, and MySQL."]
            },
            {
                'patterns': [r'project|about this chatbot'],
                'responses': ["This is an advanced rule-based chatbot with AI capabilities. It uses pattern matching for simple queries and AI for more complex questions."]
            }
        ]

# Function to match user input with rule patterns
def find_response(user_input):
    user_input = user_input.lower().strip()
    
    # Try to match with simple rules first
    for rule in st.session_state.rules:
        for pattern in rule['patterns']:
            if re.search(pattern, user_input):
                return random.choice(rule['responses'])
    
    # If no simple rule matches, use Groq API for more complex responses
    return get_ai_response(user_input)

# Function to get response from Groq API
def get_ai_response(query):
    try:
        with st.spinner("Thinking..."):
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful, friendly assistant created by Nandesh Kalashetti. Keep your answers concise, informative and engaging."},
                    {"role": "user", "content": query}
                ],
                model="llama3-8b-8192",
                max_tokens=800
            )
            return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error connecting to Groq API: {str(e)}")
        return "I'm having trouble connecting to my AI capabilities right now. Please try again later."

# Function to add to chat history
def add_to_chat_history(query):
    if query not in [item[0] for item in st.session_state.chat_history]:
        st.session_state.chat_history.insert(0, (query, time.time()))
        # Keep only the most recent 15 queries
        if len(st.session_state.chat_history) > 15:
            st.session_state.chat_history = st.session_state.chat_history[:15]

# Main app function
def main():
    load_css()
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        # Developer info with profile picture
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("""
        <div class="logo-container">
            <img src="https://nandeshkalashetti.netlify.app/img/person.jpg" alt="Nandesh Kalashetti">
            <h2 style="margin-top: 10px; color: #f8fafc; font-size: 18px;">Nandesh Kalashetti</h2>
            <p style="color: #94a3b8; font-size: 13px;">Full-Stack Developer</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Connect links
        st.markdown("<h3 style='color: #f8fafc; font-size: 15px; margin: 15px 0 10px;'>Connect With Me</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="connect-links">
            <a href="https://github.com/Universe7Nandu" target="_blank" class="connect-link">
                <img src="https://img.icons8.com/fluent/24/000000/github.png" width="18" />
            </a>
            <a href="https://www.linkedin.com/in/nandesh-kalashetti-333a78250/" target="_blank" class="connect-link">
                <img src="https://img.icons8.com/color/24/000000/linkedin.png" width="18" />
            </a>
            <a href="https://twitter.com/UniverseMath25" target="_blank" class="connect-link">
                <img src="https://img.icons8.com/color/24/000000/twitter--v1.png" width="18" />
            </a>
            <a href="https://www.instagram.com/nandesh_kalshetti/" target="_blank" class="connect-link">
                <img src="https://img.icons8.com/fluency/24/000000/instagram-new.png" width="18" />
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat history in sidebar
        if st.session_state.chat_history:
            st.markdown("<h3 style='color: #f8fafc; font-size: 15px; margin: 20px 0 5px;'>Recent Chats</h3>", unsafe_allow_html=True)
            st.markdown('<div class="history-container">', unsafe_allow_html=True)
            
            for i, (query, timestamp) in enumerate(st.session_state.chat_history):
                if st.markdown(f"""
                <div class="history-item" onclick="this.classList.toggle('active');">
                    {query[:40] + '...' if len(query) > 40 else query}
                </div>
                """, unsafe_allow_html=True):
                    # If clicked, populate with this query
                    st.session_state.prefill_query = query
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature cards
        st.markdown("<h3 style='color: #f8fafc; font-size: 15px; margin: 20px 0 5px;'>Features</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #f8fafc; font-size: 14px; margin-bottom: 5px;">🧠 Rule-Based Intelligence</h4>
            <p style="color: #94a3b8; font-size: 12px;">Pattern matching for quick responses</p>
        </div>
        
        <div class="feature-card">
            <h4 style="color: #f8fafc; font-size: 14px; margin-bottom: 5px;">🤖 AI Integration</h4>
            <p style="color: #94a3b8; font-size: 12px;">Powered by Groq LLM API</p>
        </div>
        
        <div class="feature-card">
            <h4 style="color: #f8fafc; font-size: 14px; margin-bottom: 5px;">💬 Natural Conversations</h4>
            <p style="color: #94a3b8; font-size: 12px;">Context-aware dialogue</p>
        </div>
        
        <div class="feature-card">
            <h4 style="color: #f8fafc; font-size: 14px; margin-bottom: 5px;">📱 Responsive Design</h4>
            <p style="color: #94a3b8; font-size: 12px;">Works on all devices</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tech stack
        st.markdown("<h3 style='color: #f8fafc; font-size: 15px; margin: 20px 0 5px;'>Tech Stack</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 5px;">
            <span class="tech-badge">Python</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Groq API</span>
            <span class="tech-badge">LLaMA 3</span>
            <span class="tech-badge">CSS3</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content - simplified structure
    st.markdown('<div class="chat-interface">', unsafe_allow_html=True)
    
    # Chat title
    st.markdown("""
    <div class="chat-title">
        <h1>✨ AI Assistant with Advanced Intelligence</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Messages container
    st.markdown('<div class="messages-container">', unsafe_allow_html=True)
    
    # Display welcome message if no messages yet
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-container">
            <img src="https://img.icons8.com/fluency/96/000000/chatbot.png" style="width: 80px; margin-bottom: 20px;" alt="Chatbot Icon">
            <h3 style="color: #f8fafc; font-size: 24px; margin-bottom: 10px;">Welcome to the AI Assistant!</h3>
            <p style="color: #94a3b8; font-size: 16px;">Start the conversation by typing a message below.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Using columns for better layout of input and button
    col1, col2 = st.columns([6, 1])
    
    with col1:
        # Check if we need to prefill from history
        prefill = st.session_state.get('prefill_query', '')
        if prefill:
            user_input = st.text_input(
                label="Message",
                value=prefill,
                placeholder="Type your message here...",
                key="user_input", 
                label_visibility="collapsed"
            )
            # Clear prefill after setting it
            st.session_state.prefill_query = ''
        else:
            user_input = st.text_input(
                label="Message",
                placeholder="Type your message here...",
                key="user_input", 
                label_visibility="collapsed"
            )
    
    with col2:
        send_button = st.button("Send", key="send")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        © 2025 Nandesh Kalashetti | Advanced Rule-Based Chatbot with AI Capabilities
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-interface
    
    # Process user input
    if send_button and user_input:
        # Add to chat history
        add_to_chat_history(user_input)
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        response = find_response(user_input)
        
        # Add bot response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Use st.rerun() instead of experimental_rerun
        st.rerun()

if __name__ == "__main__":
    main()
