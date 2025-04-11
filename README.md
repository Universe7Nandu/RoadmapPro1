# 🤖 AI Assistant with Rule-Based Intelligence

<div align="center">
  <img src="https://raw.githubusercontent.com/Universe7Nandu/Rule-Based-Chatbot/main/.github/banner.png" alt="AI Assistant Banner" width="800px" style="border-radius: 10px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);">

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Groq%20API-0D96F6?style=for-the-badge&logo=openai&logoColor=white" alt="Groq API">
    <img src="https://img.shields.io/badge/LLaMA%203-5436DA?style=for-the-badge&logo=meta&logoColor=white" alt="LLaMA 3">
    <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3">
  </p>
  
  <p align="center">
    <a href="https://universe7nandu-rule-based-chatbot-app-jgrqn3.streamlit.app/">View Demo</a>
    ·
    <a href="https://github.com/Universe7Nandu/Rule-Based-Chatbot/issues">Report Bug</a>
    ·
    <a href="https://github.com/Universe7Nandu/Rule-Based-Chatbot/issues">Request Feature</a>
  </p>
</div>

## ✨ Overview

AI Assistant is a sophisticated chatbot that combines rule-based pattern matching with advanced AI capabilities to deliver a responsive and intelligent conversational experience. Built with performance and user experience in mind, it features a beautiful modern UI that works seamlessly across desktop and mobile devices.

### 📱 Live Demo

Experience the AI Assistant in action: [Live Demo](https://universe7nandu-rule-based-chatbot-app-jgrqn3.streamlit.app/)

<div align="center">
  <img src="https://raw.githubusercontent.com/Universe7Nandu/Rule-Based-Chatbot/main/.github/demo.gif" alt="AI Assistant Demo" width="700px" style="border-radius: 8px; margin: 20px 0; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);">
</div>

## 🎯 Key Features

- **Rule-Based Intelligence**: Pattern matching using regex provides instant responses to common queries
- **AI Capabilities**: Powered by Groq API with LLaMA 3 model for handling complex questions
- **Responsive Design**: Optimized UI that works beautifully on desktops, tablets, and mobile devices
- **Chat History**: Remembers your recent conversations for easy reference
- **Modern UI**: Sleek design with glassmorphism effects, animations, and intuitive interaction
- **Fast Response Time**: Optimized for performance with minimal latency

## 🧠 Why This Matters

In today's fast-paced digital world, immediate access to information is crucial. This AI Assistant represents an important step in creating accessible AI tools that:

1. **Reduce Response Time**: By combining pre-defined rules with AI, users get instant answers to common questions
2. **Lower Computational Costs**: Rule-based systems handle simple queries efficiently without needing to call costly API endpoints
3. **Increase Accessibility**: The responsive design ensures that AI assistance is available on any device
4. **Improve User Experience**: Beautiful UI and natural conversation flow make interacting with AI more engaging

## 🛠️ Tech Stack

- **Frontend**: Streamlit, CSS3, HTML5
- **Backend**: Python
- **AI**: Groq API, LLaMA 3 model
- **Pattern Matching**: Regular Expressions (Regex)
- **Hosting**: Streamlit Cloud

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Universe7Nandu/Rule-Based-Chatbot.git
cd Rule-Based-Chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

## 🚀 Usage

The AI Assistant can:

- Answer common questions instantly using pattern matching
- Provide detailed responses to complex queries using AI
- Remember your conversation history
- Adapt to different devices with responsive design

To customize the rule patterns, modify the `rules` list in the `init_session_state()` function in `app.py`.

## 🔍 How It Works

The chatbot uses a hybrid approach to handle different types of queries:

1. **Rule-Based Processing**:
   - User input is matched against predefined regex patterns
   - If a match is found, a pre-written response is returned instantly

2. **AI Processing**:
   - If no rule matches, the query is sent to the Groq API
   - The LLaMA 3 model generates a contextually appropriate response
   - The response is displayed to the user

3. **Chat History Management**:
   - Recent queries are stored in the session state
   - Users can revisit previous conversations easily

## 🤝 Connect with the Developer

<div align="center">
  <a href="https://nandeshkalashetti.netlify.app/">
    <img src="https://img.shields.io/badge/Portfolio-3D3D3D?style=for-the-badge&logo=firefox&logoColor=white" alt="Portfolio">
  </a>
  <a href="https://github.com/Universe7Nandu">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/nandesh-kalashetti-333a78250/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://twitter.com/UniverseMath25">
    <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
  </a>
  <a href="https://www.instagram.com/nandesh_kalshetti/">
    <img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram">
  </a>
</div>

## 📝 Future Enhancements

- Multi-language support
- Voice input and output
- Integration with external APIs for real-time data
- User authentication for personalized experiences
- Expanded rule set for more efficient responses

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>
    <b>Designed and Developed with ❤️ by <a href="https://nandeshkalashetti.netlify.app/">Nandesh Kalashetti</a></b>
  </p>
  <p>
    <i>If you found this project helpful or interesting, please consider giving it a ⭐!</i>
  </p>
</div> 