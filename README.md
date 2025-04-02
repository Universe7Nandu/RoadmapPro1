# TravelPro - AI Travel Planning Platform

TravelPro is a sophisticated travel planning platform that delivers Google-quality search experiences with personalized itinerary generation. This enterprise solution optimizes travel recommendations through advanced AI algorithms.

## Features

- **Smart Search**: Find perfect destinations matching your preferences with advanced search capabilities
- **Personalized Itineraries**: Generate customized travel plans based on your interests, budget, and timeframe
- **Travel Assistant**: Get expert travel advice through an AI-powered chatbot
- **Destination Explorer**: Browse detailed information about popular travel destinations
- **Interactive UI**: Modern, responsive design with intuitive navigation

## Technical Details

TravelPro leverages several advanced technologies:

- Streamlit for the frontend interface
- LangChain for AI orchestration
- Groq API for fast, high-quality AI responses
- ChromaDB for vector storage and semantic search
- HuggingFace embeddings for natural language understanding

## Deployment Instructions

### Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Sign up on [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select this repository
4. Set the main file as `app.py`
5. Add the following secrets in Streamlit settings:
   - `GROQ_API_KEY`: Your Groq API key

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/travelpro.git
cd travelpro

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

## Credits

- Developed by [Nandesh Kalashetti](https://nandeshkalashetti.netlify.app/)
- GitHub: [@Universe7Nandu](https://github.com/Universe7Nandu)

## License

MIT License 