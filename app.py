import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import os
import json
import random
from datetime import datetime, timedelta

# Travel destinations data
TRAVEL_DATA = {
    "Delhi": {
        "highlights": ["Red Fort", "India Gate", "Qutub Minar", "Lotus Temple"],
        "best_time": "October to March",
        "budget": "₹10,000 to ₹50,000",
        "nearby": ["Agra (Taj Mahal)", "Jaipur", "Rishikesh"],
        "transport": ["Train", "Bus", "Flight"],
        "image": "https://images.unsplash.com/photo-1587474260584-136574528ed5?q=80&w=1000&auto=format&fit=crop",
        "description": "Delhi, India's capital territory, is a massive metropolitan area in the country's north. It's a city with many faces, reflecting India's long history and diverse culture."
    },
    "Mumbai": {
        "highlights": ["Gateway of India", "Marine Drive", "Elephanta Caves", "Sanjay Gandhi National Park"],
        "best_time": "November to February",
        "budget": "₹15,000 to ₹60,000",
        "nearby": ["Lonavala", "Pune", "Alibaug"],
        "transport": ["Train", "Bus", "Flight"],
        "image": "https://images.unsplash.com/photo-1570168007204-dfb528c6958f?q=80&w=1000&auto=format&fit=crop",
        "description": "Mumbai is a densely populated city on India's west coast. A financial center, it's India's largest city and home to Bollywood, one of the world's biggest film industries."
    },
    "Goa": {
        "highlights": ["Baga Beach", "Fort Aguada", "Basilica of Bom Jesus", "Dudhsagar Falls"],
        "best_time": "November to February",
        "budget": "₹8,000 to ₹40,000",
        "nearby": ["Hampi", "Mangalore", "Belgaum"],
        "transport": ["Train", "Bus", "Flight"],
        "image": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?q=80&w=1000&auto=format&fit=crop",
        "description": "Goa is a state in western India with coastlines stretching along the Arabian Sea. Its long history as a Portuguese colony prior to 1961 is evident in its preserved 17th-century churches."
    },
    "Jaipur": {
        "highlights": ["Amber Fort", "Hawa Mahal", "City Palace", "Jantar Mantar"],
        "best_time": "October to March",
        "budget": "₹7,000 to ₹35,000",
        "nearby": ["Pushkar", "Ajmer", "Ranthambore"],
        "transport": ["Train", "Bus", "Flight"],
        "image": "https://images.unsplash.com/photo-1599661046289-e31897846e41?q=80&w=1000&auto=format&fit=crop",
        "description": "Jaipur is the capital of India's Rajasthan state, known as the 'Pink City' for its trademark building color. It's part of the Golden Triangle tourist circuit along with Delhi and Agra."
    },
    "Bangalore": {
        "highlights": ["Lalbagh Botanical Garden", "Bangalore Palace", "Cubbon Park", "Bannerghatta Biological Park"],
        "best_time": "Year-round (October to February is ideal)",
        "budget": "₹10,000 to ₹45,000",
        "nearby": ["Mysore", "Coorg", "Ooty"],
        "transport": ["Train", "Bus", "Flight"],
        "image": "https://images.unsplash.com/photo-1580674684081-7617fbf3d745?q=80&w=1000&auto=format&fit=crop",
        "description": "Bangalore (officially Bengaluru) is the capital of India's southern Karnataka state. The center of India's high-tech industry, it's known for its parks and vibrant nightlife."
    }
}

# Initialize Streamlit App with custom theme
st.set_page_config(
    page_title="TravelPro | Smart Travel Planning",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        font-family: 'Roboto', sans-serif;
    }
    .destination-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .destination-card:hover {
        transform: translateY(-5px);
    }
    .highlight-text {
        color: #FF5722;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 3px;
        margin: 10px 0;
    }
    .chat-message-user {
        background-color: #E8F5E9;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-end;
        margin-left: auto;
        color: #1B5E20;
    }
    .chat-message-ai {
        background-color: #F3E5F5;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        max-width: 80%;
        align-self: flex-start;
        margin-right: auto;
        color: #4A148C;
    }
    .itinerary-day {
        background-color: #FAFAFA;
        border-left: 4px solid #9C27B0;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .feature-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .hero-section {
        padding: 40px 0;
        text-align: center;
        background: linear-gradient(135deg, #6B73FF 10%, #000DFF 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .tab-content {
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize memory and models
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChatGroq correctly
try:
    chat = ChatGroq(
        model_name="llama3-70b-8192", 
        temperature=0.7, 
        groq_api_key="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"
    )
except Exception as e:
    st.error(f"⚠️ Chat model initialization failed: {str(e)}")
    chat = None  # Prevents further errors

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="travel_knowledge_base")

def retrieve_context(query, top_k=3):
    """Retrieve relevant documents from ChromaDB using embeddings."""
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results and results["documents"] else ["No relevant context found."]

def query_llama3(user_query):
    """Query the Llama3 model with user input and retrieved context."""
    system_prompt = """
    You are 'TravelPro Assistant,' an intelligent travel planning AI. Your role is to help users:
    1. Plan detailed itineraries based on their preferences
    2. Find suitable destinations matching their criteria
    3. Suggest optimal transport options and routes
    4. Provide accurate budget estimates and travel tips
    5. Recommend attractions, activities, and hidden gems
    
    Be specific, concise, and personalized in your recommendations.
    """
    
    # First, check if query matches any destination directly
    destination_info = ""
    for dest, info in TRAVEL_DATA.items():
        if dest.lower() in user_query.lower():
            destination_info = f"Information about {dest}: {json.dumps(info, indent=2)}"
            break
    
    retrieved_context = retrieve_context(user_query)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"DB Context: {retrieved_context}\nDestination Data: {destination_info}\nQuestion: {user_query}")
    ]
    
    if not isinstance(chat, ChatGroq):  
        return user_query, "⚠️ Chat model initialization error."

    try:
        response = chat.invoke(messages)
        return user_query, response.content
    except Exception as e:
        return user_query, f"⚠️ API Error: {str(e)}"

def generate_itinerary(destination, days, interests, budget_level):
    """Generate a personalized travel itinerary."""
    if destination not in TRAVEL_DATA:
        return f"Sorry, I don't have enough information about {destination} to create an itinerary."
    
    dest_data = TRAVEL_DATA[destination]
    itinerary = []
    
    all_attractions = dest_data["highlights"] + ["Local markets", "Food tour", "Cultural show"]
    nearby = dest_data["nearby"]
    
    # Filter attractions based on interests
    filtered_attractions = all_attractions.copy()
    if "culture" in interests:
        filtered_attractions.extend(["Museum visit", "Historical sites tour"])
    if "adventure" in interests:
        filtered_attractions.extend(["Hiking trip", "Adventure sports"])
    if "food" in interests:
        filtered_attractions.extend(["Cooking class", "Street food tour"])
    if "relaxation" in interests:
        filtered_attractions.extend(["Spa day", "Beach time"])
    
    # Generate itinerary for each day
    for day in range(1, days + 1):
        daily_plan = {
            "day": day,
            "date": (datetime.now() + timedelta(days=day-1)).strftime("%d %b %Y"),
            "morning": random.choice(filtered_attractions),
            "afternoon": random.choice(filtered_attractions),
            "evening": random.choice(filtered_attractions),
            "accommodation": "Luxury hotel" if budget_level == "high" else "Mid-range hotel" if budget_level == "medium" else "Budget hostel"
        }
        
        # Add nearby destination on last day if multiple days
        if day == days and days > 2 and day > 1:
            daily_plan["day_trip"] = f"Day trip to {random.choice(nearby)}"
            
        itinerary.append(daily_plan)
    
    return itinerary

# App state management
if "page" not in st.session_state:
    st.session_state.page = "home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "destinations" not in st.session_state:
    st.session_state.destinations = list(TRAVEL_DATA.keys())
if "selected_destination" not in st.session_state:
    st.session_state.selected_destination = None
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

# Sidebar navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=80)
    st.title("TravelPro")
    st.caption("Your AI-powered travel companion")
    
    st.divider()
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
    if st.button("🔍 Explore Destinations", use_container_width=True):
        st.session_state.page = "explore"
    if st.button("📝 Plan Itinerary", use_container_width=True):
        st.session_state.page = "itinerary"
    if st.button("💬 Travel Assistant", use_container_width=True):
        st.session_state.page = "chat"
    
    st.divider()
    
    st.markdown("### Filters")
    budget_filter = st.select_slider("Budget Range", options=["Budget", "Mid-range", "Luxury"])
    season_filter = st.multiselect("Best Time to Visit", options=["Winter", "Summer", "Monsoon", "Spring", "Autumn"])
    
    st.divider()
    
    # About section
    st.markdown("### About TravelPro")
    st.markdown("TravelPro is an AI-powered travel planning platform created by [Nandesh Kalashetti](https://nandeshkalashetti.netlify.app/).")
    
    # Clear history
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Main content based on selected page
if st.session_state.page == "home":
    # Hero section
    st.markdown('<div class="hero-section"><h1>TravelPro</h1><p>Discover, Plan, and Experience Perfect Travels</p></div>', unsafe_allow_html=True)
    
    # Main features in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card"><h3>🔍 Smart Search</h3><p>Find perfect destinations matching your preferences with Google-quality search.</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card"><h3>📝 AI Itineraries</h3><p>Get personalized travel plans optimized for your time and interests.</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card"><h3>💬 Travel Assistant</h3><p>Ask any travel question and get expert recommendations instantly.</p></div>', unsafe_allow_html=True)
    
    # Featured destinations carousel
    st.markdown("### Featured Destinations")
    featured_cols = st.columns(3)
    
    for i, (dest, info) in enumerate(list(TRAVEL_DATA.items())[:3]):
        with featured_cols[i]:
            st.image(info["image"], use_container_width=True)
            st.markdown(f"**{dest}**")
            st.markdown(f"{info['description'][:100]}...")
            if st.button(f"Explore {dest}", key=f"feat_{dest}"):
                st.session_state.selected_destination = dest
                st.session_state.page = "explore"
    
    # User testimonials
    st.markdown("### What Our Users Say")
    reviews_cols = st.columns(3)
    
    with reviews_cols[0]:
        st.markdown("""
        ⭐⭐⭐⭐⭐
        "TravelPro made planning my family vacation so easy! The AI suggestions were spot on."
        - Amit S.
        """)
    
    with reviews_cols[1]:
        st.markdown("""
        ⭐⭐⭐⭐⭐
        "I discovered places I never knew existed. Best travel planning tool ever!"
        - Priya R.
        """)
    
    with reviews_cols[2]:
        st.markdown("""
        ⭐⭐⭐⭐
        "The itinerary generator saved me hours of research. Highly recommended!"
        - Rahul K.
        """)

elif st.session_state.page == "explore":
    st.title("Explore Destinations")
    
    # Search bar
    search_query = st.text_input("Search destinations, activities, or places", placeholder="Try 'beach destinations in India' or 'historical places'")
    
    if search_query:
        st.subheader(f"Search Results for: {search_query}")
        filtered_destinations = [dest for dest in TRAVEL_DATA.keys() 
                                if search_query.lower() in dest.lower() 
                                or any(item.lower() in search_query.lower() for item in TRAVEL_DATA[dest]["highlights"])
                                or search_query.lower() in TRAVEL_DATA[dest]["description"].lower()]
        
        if not filtered_destinations:
            st.info("No exact matches found. Here are some recommended destinations:")
            filtered_destinations = list(TRAVEL_DATA.keys())
    else:
        filtered_destinations = list(TRAVEL_DATA.keys())
    
    # Display destinations as cards
    for dest in filtered_destinations:
        data = TRAVEL_DATA[dest]
        
        with st.container():
            st.markdown(f'<div class="destination-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(data["image"], use_container_width=True)
            
            with col2:
                st.subheader(dest)
                st.markdown(data["description"])
                
                details_col1, details_col2 = st.columns(2)
                
                with details_col1:
                    st.markdown(f"**Best Time to Visit:** {data['best_time']}")
                    st.markdown(f"**Budget Estimate:** {data['budget']}")
                
                with details_col2:
                    st.markdown(f"**Key Highlights:** {', '.join(data['highlights'][:2])}...")
                    st.markdown(f"**Nearby Attractions:** {', '.join(data['nearby'][:2])}...")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if st.button(f"Select {dest}", key=f"select_{dest}"):
                    st.session_state.selected_destination = dest
                    st.session_state.page = "itinerary"
            with col2:
                st.button(f"View Details", key=f"details_{dest}", disabled=True)
            with col3:
                st.button(f"Add to Wishlist", key=f"wish_{dest}", disabled=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
elif st.session_state.page == "itinerary":
    st.title("Create Your Personalized Itinerary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Itinerary form
        selected_destination = st.selectbox("Select Destination", 
                                          options=list(TRAVEL_DATA.keys()),
                                          index=list(TRAVEL_DATA.keys()).index(st.session_state.selected_destination) if st.session_state.selected_destination else 0)
        
        trip_duration = st.slider("Trip Duration (Days)", min_value=1, max_value=14, value=3)
        
        interests = st.multiselect("Your Interests", 
                                 options=["Culture", "Adventure", "Food", "Nature", "Shopping", "Relaxation"],
                                 default=["Culture", "Food"])
        
        budget_level = st.select_slider("Budget Level", options=["low", "medium", "high"], value="medium")
        
        if st.button("Generate Itinerary", type="primary"):
            with st.spinner("Creating your perfect itinerary..."):
                # Generate the itinerary
                itinerary = generate_itinerary(selected_destination, trip_duration, [i.lower() for i in interests], budget_level)
                st.session_state.itinerary = itinerary
                st.session_state.selected_destination = selected_destination
    
    with col2:
        if st.session_state.selected_destination:
            dest = st.session_state.selected_destination
            data = TRAVEL_DATA[dest]
            
            st.image(data["image"], use_container_width=True)
            st.markdown(f"### About {dest}")
            st.markdown(data["description"])
            
            st.markdown("**Highlights:**")
            for highlight in data["highlights"]:
                st.markdown(f"- {highlight}")
            
            st.markdown(f"**Best Time to Visit:** {data['best_time']}")
    
    # Display generated itinerary
    if st.session_state.itinerary:
        st.markdown(f"## Your {st.session_state.selected_destination} Itinerary")
        
        for day_plan in st.session_state.itinerary:
            if isinstance(day_plan, dict):  # Ensure it's a dictionary
                with st.container():
                    st.markdown(f'<div class="itinerary-day">', unsafe_allow_html=True)
                    st.subheader(f"Day {day_plan['day']} - {day_plan['date']}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Morning**")
                        st.markdown(f"{day_plan['morning']}")
                    
                    with col2:
                        st.markdown("**Afternoon**")
                        st.markdown(f"{day_plan['afternoon']}")
                    
                    with col3:
                        st.markdown("**Evening**")
                        st.markdown(f"{day_plan['evening']}")
                    
                    st.markdown(f"**Accommodation:** {day_plan['accommodation']}")
                    
                    if 'day_trip' in day_plan:
                        st.info(f"**Special:** {day_plan['day_trip']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning(day_plan)  # Display error message
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download Itinerary (PDF)", "Itinerary PDF will be available in the full version", disabled=True)
        with col2:
            st.button("Share Itinerary", disabled=True)

elif st.session_state.page == "chat":
    st.title("TravelPro AI Assistant")
    st.markdown("Ask me anything about travel planning, destinations, budgets, or itineraries")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for chat_msg in st.session_state.chat_history:
            user_msg, bot_response = chat_msg
            st.markdown(f'<div class="chat-message-user">👤 {user_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message-ai">🤖 {bot_response}</div>', unsafe_allow_html=True)
    
    # User input
    user_query = st.chat_input("Type your travel question here...")
    if user_query:
        with st.spinner("Finding the best answer for you..."):
            user_msg, bot_response = query_llama3(user_query)
            st.session_state.chat_history.append((user_msg, bot_response))
            st.rerun()  # Refresh the app to update chat history

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #ddd;">
    <p>© 2025 TravelPro. Created by <a href="https://nandeshkalashetti.netlify.app/" target="_blank">Nandesh Kalashetti</a> | 
    <a href="https://github.com/Universe7Nandu" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
