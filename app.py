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
from PyPDF2 import PdfReader, PdfWriter
import os
import json
import random
import io
from datetime import datetime, timedelta
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch

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
        background-color: #f8f9fa;
    }
    .stApp {
        font-family: 'Poppins', sans-serif;
    }
    .destination-card {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid #f0f0f0;
    }
    .destination-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        border-color: #e1e1e1;
    }
    .highlight-text {
        color: #FF5722;
        font-weight: 600;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 20px;
        border-radius: 5px;
        margin: 15px 0;
    }
    .chat-message-user {
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
        padding: 12px 18px;
        border-radius: 18px 18px 0 18px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-end;
        margin-left: auto;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message-ai {
        background: linear-gradient(135deg, #7e57c2 0%, #5e35b1 100%);
        padding: 12px 18px;
        border-radius: 18px 18px 18px 0;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-start;
        margin-right: auto;
        color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .itinerary-day {
        background-color: white;
        border-left: 5px solid #ff6d00;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        height: 100%;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
        border: 1px solid #f0f0f0;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .hero-section {
        padding: 60px 20px;
        text-align: center;
        background: linear-gradient(135deg, #3f51b5 0%, #2196f3 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    .tab-content {
        padding: 25px 0;
    }
    
    .stButton>button {
        border-radius: 25px;
        font-weight: 500;
        border: none;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sidebar .stButton>button {
        background: linear-gradient(135deg, #4a4af8 0%, #2a2af0 100%);
        color: white;
        border-radius: 12px;
    }
    
    .select-button {
        background: linear-gradient(135deg, #ff6d00 0%, #ff9100 100%);
        color: white;
        border-radius: 25px;
        padding: 8px 16px;
        font-weight: 500;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .select-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    h1, h2, h3 {
        color: #333;
        font-weight: 600;
    }
    
    .testimonial-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .search-bar {
        border-radius: 30px;
        padding: 10px 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
        padding: 30px;
        border-radius: 10px;
        margin-top: 50px;
        text-align: center;
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div > div {
        background-color: #4a4af8;
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

def create_itinerary_pdf(destination, itinerary):
    """Create a PDF document for the itinerary."""
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create a custom style for headings
    styles.add(ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#3f51b5"),
        spaceAfter=20
    ))
    
    styles.add(ParagraphStyle(
        name='SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor("#ff6d00"),
        spaceAfter=15
    ))
    
    styles.add(ParagraphStyle(
        name='DayStyle',
        parent=styles['Heading3'],
        fontSize=16,
        textColor=colors.HexColor("#2196f3"),
        spaceAfter=10
    ))
    
    story = []
    
    # Add title
    title = Paragraph(f"Your {destination} Itinerary", styles['TitleStyle'])
    story.append(title)
    
    # Add subtitle with date range
    if itinerary and isinstance(itinerary[0], dict):
        start_date = itinerary[0]["date"]
        end_date = itinerary[-1]["date"]
        subtitle = Paragraph(f"Travel dates: {start_date} to {end_date}", styles['SubtitleStyle'])
        story.append(subtitle)
    
    story.append(Spacer(1, 0.25*inch))
    
    # Add destination info
    if destination in TRAVEL_DATA:
        dest_data = TRAVEL_DATA[destination]
        story.append(Paragraph(f"About {destination}", styles['SubtitleStyle']))
        story.append(Paragraph(dest_data["description"], styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add highlights
        story.append(Paragraph("Key Highlights:", styles['Heading4']))
        for highlight in dest_data["highlights"]:
            story.append(Paragraph(f"• {highlight}", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Best Time to Visit: {dest_data['best_time']}", styles['Normal']))
        story.append(Paragraph(f"Budget Estimate: {dest_data['budget']}", styles['Normal']))
        
        story.append(Spacer(1, 0.5*inch))
    
    # Add daily itinerary
    story.append(Paragraph("Your Day-by-Day Itinerary", styles['SubtitleStyle']))
    
    for day_plan in itinerary:
        if isinstance(day_plan, dict):
            # Add day header
            day_title = Paragraph(f"Day {day_plan['day']} - {day_plan['date']}", styles['DayStyle'])
            story.append(day_title)
            
            # Create a table for the day's activities
            data = [
                ["Time", "Activity"],
                ["Morning", day_plan["morning"]],
                ["Afternoon", day_plan["afternoon"]],
                ["Evening", day_plan["evening"]]
            ]
            
            table = Table(data, colWidths=[1.5*inch, 4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#e3f2fd")),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor("#0d47a1")),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor("#f5f5f5")),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.1*inch))
            
            # Add accommodation
            story.append(Paragraph(f"Accommodation: {day_plan['accommodation']}", styles['Normal']))
            
            # Add day trip if available
            if 'day_trip' in day_plan:
                story.append(Paragraph(f"Special: {day_plan['day_trip']}", styles['Normal']))
            
            story.append(Spacer(1, 0.3*inch))
        else:
            # Handle error message case
            story.append(Paragraph(day_plan, styles['Normal']))
    
    # Add footer with branding
    story.append(Spacer(1, 0.5*inch))
    footer_text = "Created with ❤️ by TravelPro | Your AI-powered travel companion"
    footer = Paragraph(footer_text, styles['Italic'])
    story.append(footer)
    
    # Build the PDF
    doc.build(story)
    
    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

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
if "itinerary_pdf" not in st.session_state:
    st.session_state.itinerary_pdf = None

# Sidebar navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=80)
    st.title("TravelPro")
    st.caption("Your AI-powered travel companion")
    
    st.divider()
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    if st.button("🔍 Explore Destinations", use_container_width=True):
        st.session_state.page = "explore"
        st.rerun()
    if st.button("📝 Plan Itinerary", use_container_width=True):
        st.session_state.page = "itinerary"
        st.rerun()
    if st.button("💬 Travel Assistant", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()
    
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
    # Hero section with improved styling
    st.markdown('''
    <div class="hero-section">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem;">TravelPro</h1>
        <p style="font-size: 1.5rem; margin-bottom: 2rem;">Discover, Plan, and Experience Perfect Travels</p>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 25px;">
            <button style="background: white; color: #3f51b5; border: none; border-radius: 30px; padding: 12px 25px; font-weight: 600; cursor: pointer; transition: all 0.3s ease;">Start Planning</button>
            <button style="background: rgba(255,255,255,0.15); color: white; border: 1px solid white; border-radius: 30px; padding: 12px 25px; font-weight: 600; cursor: pointer; transition: all 0.3s ease;">Learn More</button>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Main features in columns with improved styling
    st.markdown("### How TravelPro Works", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="feature-card">
            <div style="text-align: center; margin-bottom: 15px;">
                <img src="https://cdn-icons-png.flaticon.com/512/4300/4300059.png" width="80">
            </div>
            <h3 style="text-align: center; color: #3f51b5;">🔍 Smart Search</h3>
            <p style="text-align: center;">Find perfect destinations matching your preferences with Google-quality search.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="feature-card">
            <div style="text-align: center; margin-bottom: 15px;">
                <img src="https://cdn-icons-png.flaticon.com/512/2702/2702134.png" width="80">
            </div>
            <h3 style="text-align: center; color: #3f51b5;">📝 AI Itineraries</h3>
            <p style="text-align: center;">Get personalized travel plans optimized for your time and interests.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div class="feature-card">
            <div style="text-align: center; margin-bottom: 15px;">
                <img src="https://cdn-icons-png.flaticon.com/512/3095/3095583.png" width="80">
            </div>
            <h3 style="text-align: center; color: #3f51b5;">💬 Travel Assistant</h3>
            <p style="text-align: center;">Ask any travel question and get expert recommendations instantly.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Stats section
    st.markdown('''
    <div style="display: flex; justify-content: space-between; margin: 40px 0; text-align: center;">
        <div>
            <h2 style="color: #3f51b5; font-weight: 700; margin-bottom: 10px;">500+</h2>
            <p style="color: #666;">Destinations</p>
        </div>
        <div>
            <h2 style="color: #3f51b5; font-weight: 700; margin-bottom: 10px;">20K+</h2>
            <p style="color: #666;">Happy Travelers</p>
        </div>
        <div>
            <h2 style="color: #3f51b5; font-weight: 700; margin-bottom: 10px;">98%</h2>
            <p style="color: #666;">Satisfaction Rate</p>
        </div>
        <div>
            <h2 style="color: #3f51b5; font-weight: 700; margin-bottom: 10px;">24/7</h2>
            <p style="color: #666;">Support</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Featured destinations carousel with improved styling
    st.markdown("<h3 style='margin-top: 40px;'>Featured Destinations</h3>", unsafe_allow_html=True)
    featured_cols = st.columns(3)
    
    for i, (dest, info) in enumerate(list(TRAVEL_DATA.items())[:3]):
        with featured_cols[i]:
            st.markdown(f'''
            <div style="background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 5px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease;">
                <div style="height: 200px; overflow: hidden;">
                    <img src="{info['image']}" style="width: 100%; height: 100%; object-fit: cover;">
                </div>
                <div style="padding: 20px;">
                    <h4 style="color: #3f51b5; margin-bottom: 10px;">{dest}</h4>
                    <p style="color: #666; font-size: 0.9rem; height: 80px; overflow: hidden;">{info['description'][:100]}...</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            if st.button(f"Explore {dest}", key=f"feat_{dest}"):
                st.session_state.selected_destination = dest
                st.session_state.page = "explore"
                st.rerun()
    
    # User testimonials with improved styling
    st.markdown("<h3 style='margin-top: 40px;'>What Our Users Say</h3>", unsafe_allow_html=True)
    reviews_cols = st.columns(3)
    
    with reviews_cols[0]:
        st.markdown('''
        <div class="testimonial-card">
            <div style="color: #ffc107; margin-bottom: 10px;">⭐⭐⭐⭐⭐</div>
            <p style="font-style: italic; color: #555;">"TravelPro made planning my family vacation so easy! The AI suggestions were spot on and saved us hours of research."</p>
            <div style="display: flex; align-items: center; margin-top: 15px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: #e1e1e1; margin-right: 10px;"></div>
                <div>
                    <p style="margin: 0; font-weight: 600;">Amit S.</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #777;">Delhi, India</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with reviews_cols[1]:
        st.markdown('''
        <div class="testimonial-card">
            <div style="color: #ffc107; margin-bottom: 10px;">⭐⭐⭐⭐⭐</div>
            <p style="font-style: italic; color: #555;">"I discovered places I never knew existed. Best travel planning tool ever! The PDF itinerary was so helpful during our trip."</p>
            <div style="display: flex; align-items: center; margin-top: 15px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: #e1e1e1; margin-right: 10px;"></div>
                <div>
                    <p style="margin: 0; font-weight: 600;">Priya R.</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #777;">Mumbai, India</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with reviews_cols[2]:
        st.markdown('''
        <div class="testimonial-card">
            <div style="color: #ffc107; margin-bottom: 10px;">⭐⭐⭐⭐</div>
            <p style="font-style: italic; color: #555;">"The itinerary generator saved me hours of research. Highly recommended! Can't wait to use it for my next vacation."</p>
            <div style="display: flex; align-items: center; margin-top: 15px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: #e1e1e1; margin-right: 10px;"></div>
                <div>
                    <p style="margin: 0; font-weight: 600;">Rahul K.</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #777;">Bangalore, India</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

elif st.session_state.page == "explore":
    st.title("Explore Destinations")
    
    # Search bar with improved styling
    search_query = st.text_input("Search destinations, activities, or places", 
                               placeholder="Try 'beach destinations in India' or 'historical places'")
    
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
                st.image(data["image"], use_column_width=True)
            
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
                    st.rerun()
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
                
                # Generate PDF
                if isinstance(itinerary, list):
                    pdf_data = create_itinerary_pdf(selected_destination, itinerary)
                    st.session_state.itinerary_pdf = pdf_data
    
    with col2:
        if st.session_state.selected_destination:
            dest = st.session_state.selected_destination
            data = TRAVEL_DATA[dest]
            
            st.image(data["image"], use_column_width=True)
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
            if st.session_state.itinerary_pdf:
                # Create a download button for the PDF
                st.download_button(
                    label="Download Itinerary PDF",
                    data=st.session_state.itinerary_pdf,
                    file_name=f"{st.session_state.selected_destination}_Itinerary.pdf",
                    mime="application/pdf",
                )
        with col2:
            st.button("Share Itinerary", disabled=True)

elif st.session_state.page == "chat":
    st.title("TravelPro AI Assistant")
    
    # Enhanced chat interface
    st.markdown('''
    <div style="background: linear-gradient(135deg, #8e24aa 0%, #3949ab 100%); padding: 20px; border-radius: 15px; color: white; margin-bottom: 30px;">
        <h2 style="margin-top: 0;">Your Personal Travel Assistant</h2>
        <p>Ask me anything about travel planning, destinations, budgets, or itineraries! I'm here to help make your next journey unforgettable.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("<p>Need inspiration? Try asking:</p>", unsafe_allow_html=True)
        
        question_cols = st.columns(2)
        with question_cols[0]:
            if st.button("What's the best time to visit Goa?"):
                user_msg, bot_response = query_llama3("What's the best time to visit Goa?")
                st.session_state.chat_history.append((user_msg, bot_response))
                st.rerun()
            
            if st.button("How much should I budget for a week in Delhi?"):
                user_msg, bot_response = query_llama3("How much should I budget for a week in Delhi?")
                st.session_state.chat_history.append((user_msg, bot_response))
                st.rerun()
        
        with question_cols[1]:
            if st.button("What are must-see attractions in Mumbai?"):
                user_msg, bot_response = query_llama3("What are must-see attractions in Mumbai?")
                st.session_state.chat_history.append((user_msg, bot_response))
                st.rerun()
            
            if st.button("Suggest a 3-day itinerary for Jaipur"):
                user_msg, bot_response = query_llama3("Suggest a 3-day itinerary for Jaipur")
                st.session_state.chat_history.append((user_msg, bot_response))
                st.rerun()
    
    # Display chat history with improved styling
    chat_container = st.container()
    with chat_container:
        for chat_msg in st.session_state.chat_history:
            user_msg, bot_response = chat_msg
            st.markdown(f'<div class="chat-message-user">👤 {user_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message-ai">🤖 {bot_response}</div>', unsafe_allow_html=True)
    
    # User input with enhanced styling
    st.markdown('''
    <div style="background: white; border-radius: 15px; padding: 5px; margin-top: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
    </div>
    ''', unsafe_allow_html=True)
    
    user_query = st.chat_input("Type your travel question here...")
    if user_query:
        with st.spinner("Finding the best answer for you..."):
            user_msg, bot_response = query_llama3(user_query)
            st.session_state.chat_history.append((user_msg, bot_response))
            st.rerun()  # Refresh the app to update chat history

# Enhanced footer
st.markdown("""
<div style="margin-top: 60px; padding-top: 40px; border-top: 1px solid #e0e0e0;">
    <div style="display: flex; justify-content: center; align-items: flex-start; flex-wrap: wrap; gap: 40px; margin-bottom: 30px; text-align: left;">
        <div>
            <h4 style="color: #3f51b5; font-weight: 600; margin-bottom: 15px; font-size: 18px;">Explore</h4>
            <p style="margin: 8px 0; color: #555; font-size: 14px; cursor: pointer;">Destinations</p>
            <p style="margin: 8px 0; color: #555; font-size: 14px; cursor: pointer;">Itineraries</p>
            <p style="margin: 8px 0; color: #555; font-size: 14px; cursor: pointer;">Travel Guides</p>
        </div>
        <div>
            <h4 style="color: #3f51b5; font-weight: 600; margin-bottom: 15px; font-size: 18px;">Company</h4>
            <p style="margin: 8px 0; color: #555; font-size: 14px; cursor: pointer;">About Us</p>
            <p style="margin: 8px 0; color: #555; font-size: 14px; cursor: pointer;">Careers</p>
            <p style="margin: 8px 0; color: #555; font-size: 14px; cursor: pointer;">Contact</p>
        </div>
        <div>
            <h4 style="color: #3f51b5; font-weight: 600; margin-bottom: 15px; font-size: 18px;">Legal</h4>
            <p style="margin: 8px 0; color: #555; font-size: 14px; cursor: pointer;">Terms of Service</p>
            <p style="margin: 8px 0; color: #555; font-size: 14px; cursor: pointer;">Privacy Policy</p>
            <p style="margin: 8px 0; color: #555; font-size: 14px; cursor: pointer;">Cookie Policy</p>
        </div>
    </div>
    <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eaeaea;">
        <p style="color: #777; font-size: 14px;">© 2025 TravelPro. Created by 
        <a href="https://nandeshkalashetti.netlify.app/" target="_blank" style="color: #3f51b5; text-decoration: none; font-weight: 500;">Nandesh Kalashetti</a> | 
        <a href="https://github.com/Universe7Nandu" target="_blank" style="color: #3f51b5; text-decoration: none; font-weight: 500;">GitHub</a></p>
    </div>
</div>
""", unsafe_allow_html=True)
