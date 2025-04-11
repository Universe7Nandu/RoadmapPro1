import pytest
import re
import os
from unittest.mock import patch, MagicMock

# Import functions from app.py
# We need to mock streamlit
with patch('streamlit.session_state', new_callable=dict):
    import app

# Test rule pattern matching
def test_pattern_matching():
    # Setup test environment
    with patch('streamlit.session_state', {'rules': [
        {
            'patterns': [r'hello|hi|hey|greetings', r'^hi$'],
            'responses': ["Hello! How can I help you today?", "Hi there! What can I assist you with?"]
        },
        {
            'patterns': [r'who are you|what are you|tell me about yourself'],
            'responses': ["I'm a rule-based chatbot with AI capabilities."]
        },
    ]}):
        # Test hello pattern
        with patch('random.choice', return_value="Hello! How can I help you today?"):
            response = app.find_response("hello")
            assert response == "Hello! How can I help you today?"
        
        # Test "who are you" pattern
        with patch('random.choice', return_value="I'm a rule-based chatbot with AI capabilities."):
            response = app.find_response("who are you")
            assert response == "I'm a rule-based chatbot with AI capabilities."
        
        # Test that unknown queries are handled by AI
        with patch('app.get_ai_response', return_value="This is an AI response"):
            response = app.find_response("something completely random")
            assert response == "This is an AI response"

# Test AI fallback
def test_ai_fallback():
    # Mock Groq API response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a mock AI response"
    
    # Test successful API call
    with patch('streamlit.spinner', MagicMock()):
        with patch('groq.Groq.chat.completions.create', return_value=mock_response):
            response = app.get_ai_response("complex query")
            assert response == "This is a mock AI response"
    
    # Test API error handling
    with patch('streamlit.spinner', MagicMock()):
        with patch('groq.Groq.chat.completions.create', side_effect=Exception("API Error")):
            with patch('streamlit.error', MagicMock()):
                response = app.get_ai_response("complex query")
                assert "I'm having trouble connecting" in response

# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", "test_chatbot.py"]) 