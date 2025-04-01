import streamlit as st
import os
import json
import datetime
import random
from datetime import timedelta
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import requests
from PIL import Image
from io import BytesIO
import base64
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import plotly.express as px
import time

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Set page configuration
st.set_page_config(
    page_title="Travel Planner Pro",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main layout styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .css-1r6slb0 {
        border-radius: 15px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1e3a8a !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    h1 {
        font-weight: 700 !important;
        font-size: 2.5rem !important;
    }
    
    h2 {
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        background-color: #4f46e5 !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background-color: #4338ca !important;
        transform: translateY(-2px) !important;
    }
    
    /* Input field styling */
    .stTextInput input, .stSelectbox, .stDateInput input {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Card container styling */
    .card-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Travel itinerary styling */
    .day-header {
        background-color: #e0f2fe;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .activity {
        padding: 0.75rem;
        border-left: 3px solid #3b82f6;
        background-color: #f8fafc;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
    }
    
    .time {
        color: #64748b;
        font-weight: 500;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 16px;
        background-color: #f1f5f9;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def get_destination_image(destination_name):
    """Fetch a relevant image for the destination using Unsplash API"""
    try:
        # Using Unsplash API (you'll need to add your Unsplash API key to .env)
        unsplash_key = os.getenv("UNSPLASH_API_KEY")
        if not unsplash_key:
            return None
            
        url = f"https://api.unsplash.com/photos/random"
        headers = {
            "Authorization": f"Client-ID {unsplash_key}",
            "Accept-Version": "v1"
        }
        params = {
            "query": destination_name,
            "orientation": "landscape"
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return data["urls"]["regular"]
    except:
        return None

def format_currency(amount):
    """Format currency with proper symbol and commas"""
    return f"${amount:,.2f}"

# Popular destinations database (simplified)
destinations = {
    "Paris": {
        "country": "France",
        "description": "The City of Light famous for the Eiffel Tower, Louvre Museum, and exquisite cuisine.",
        "attractions": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral", "Montmartre", "Champs-Élysées"],
        "image": "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80",
        "best_season": "Spring, Fall",
        "budget_range": {
            "Budget": 100,
            "Mid-range": 200,
            "Luxury": 400,
            "Ultra-Luxury": 800
        }
    },
    "Tokyo": {
        "country": "Japan",
        "description": "A bustling metropolis blending ultramodern and traditional, from neon-lit skyscrapers to historic temples.",
        "attractions": ["Tokyo Tower", "Shinjuku Gyoen", "Meiji Shrine", "Shibuya Crossing", "Asakusa Temple"],
        "image": "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1074&q=80",
        "best_season": "Spring, Fall",
        "budget_range": {
            "Budget": 120,
            "Mid-range": 250,
            "Luxury": 500,
            "Ultra-Luxury": 1000
        }
    },
    "New York": {
        "country": "USA",
        "description": "The Big Apple offers iconic skyscrapers, diverse neighborhoods, world-class museums, and Broadway shows.",
        "attractions": ["Empire State Building", "Central Park", "Times Square", "Statue of Liberty", "Metropolitan Museum of Art"],
        "image": "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80",
        "best_season": "Spring, Fall",
        "budget_range": {
            "Budget": 150,
            "Mid-range": 300,
            "Luxury": 600,
            "Ultra-Luxury": 1200
        }
    },
    "Bali": {
        "country": "Indonesia",
        "description": "Island paradise known for beautiful beaches, volcanic mountains, unique cultural heritage, and spiritual retreats.",
        "attractions": ["Ubud Monkey Forest", "Tanah Lot Temple", "Tegallalang Rice Terraces", "Uluwatu Temple", "Kuta Beach"],
        "image": "https://images.unsplash.com/photo-1537996194471-e657df975ab4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1038&q=80",
        "best_season": "Dry Season (April-October)",
        "budget_range": {
            "Budget": 50,
            "Mid-range": 100,
            "Luxury": 200,
            "Ultra-Luxury": 400
        }
    },
    "Rome": {
        "country": "Italy",
        "description": "The Eternal City with thousands of years of history, art, and delicious cuisine at every corner.",
        "attractions": ["Colosseum", "Vatican Museums", "Trevi Fountain", "Roman Forum", "Pantheon"],
        "image": "https://images.unsplash.com/photo-1552832230-c0197dd311b5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1096&q=80",
        "best_season": "Spring, Fall",
        "budget_range": {
            "Budget": 100,
            "Mid-range": 200,
            "Luxury": 400,
            "Ultra-Luxury": 800
        }
    }
}

# Sidebar content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/airplane-mode-on.png", width=100)
    st.title("Travel Planner Pro")
    st.markdown("---")
    st.markdown("### Your AI Travel Assistant")
    st.markdown("Plan your perfect vacation with AI-powered recommendations and personalized itineraries.")
    st.markdown("---")
    
    # Quick destination explorer
    st.subheader("Popular Destinations")
    for idx, (city, info) in enumerate(list(destinations.items())[:3]):
        with st.container():
            st.markdown(f"**{city}, {info['country']}**")
            st.image(info["image"], use_column_width=True, output_format="JPEG")
            st.markdown(f"*Best time: {info['best_season']}*")
            st.markdown("---")
    
    st.markdown("Made with ❤️ by TravelGenius Team")

# Main content
st.title("✈️ Discover Your Next Adventure")
st.markdown("### AI-Powered Travel Planning Made Simple")

# Tab-based interface
tab1, tab2, tab3 = st.tabs(["🔍 Explore", "🗺️ Plan Itinerary", "💡 Travel Inspiration"])

with tab1:
    st.markdown("## Find Your Perfect Destination")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        destination = st.text_input("Where do you want to go?", placeholder="City, Country or Region")
    
    with col2:
        travel_style = st.selectbox(
            "Travel Style",
            ["Any", "Adventure", "Relaxation", "Cultural", "Food & Culinary", "Nature & Wildlife", "Urban Exploration"]
        )
    
    with col3:
        budget_range = st.select_slider(
            "Budget Range",
            options=["Budget", "Mid-range", "Luxury", "Ultra-Luxury"]
        )
    
    search_button = st.button("Search Destinations", key="search_button")
    
    if search_button or destination:
        # Show loading animation
        with st.spinner("Finding the best destinations for you..."):
            # If search has specific destination
            if destination:
                # Use Groq API to generate personalized recommendations
                prompt = f"""
                Generate travel recommendations for {destination} with these preferences:
                - Travel style: {travel_style if travel_style != 'Any' else 'Not specified'}
                - Budget: {budget_range}
                
                Format the response as a Python dictionary with these keys:
                - destination_name: The main destination
                - country: Country of the destination
                - description: A captivating 2-sentence description 
                - highlights: List of 5 must-see attractions or experiences
                - best_time_to_visit: Best season or months to visit
                - budget_estimate: Estimated daily budget in USD for {budget_range} level
                - travel_tips: 3 specific travel tips for this destination
                
                Ensure it's valid Python dictionary format.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="mixtral-8x7b-32768",
                        messages=[
                            {"role": "system", "content": "You are a travel expert AI providing detailed, factual travel information."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1000
                    )
                    
                    result = response.choices[0].message.content
                    # Extract the dictionary from the response
                    try:
                        import re
                        # Try to extract dictionary using regex pattern
                        dict_pattern = r"\{[\s\S]*\}"
                        dict_match = re.search(dict_pattern, result)
                        if dict_match:
                            result_str = dict_match.group()
                            destination_info = eval(result_str)
                        else:
                            # Fallback to using a predefined data structure
                            destination_info = {
                                "destination_name": destination.title(),
                                "country": "Not specified",
                                "description": "A beautiful destination waiting to be explored.",
                                "highlights": ["Local attractions", "Cultural experiences", "Natural beauty", "Cuisine", "Historical sites"],
                                "best_time_to_visit": "Varies by season",
                                "budget_estimate": f"Varies based on {budget_range} preferences",
                                "travel_tips": ["Research local customs", "Check visa requirements", "Book accommodations in advance"]
                            }
                    except:
                        # If evaluation fails, use fallback data
                        destination_info = {
                            "destination_name": destination.title(),
                            "country": "Not specified",
                            "description": "A beautiful destination waiting to be explored.",
                            "highlights": ["Local attractions", "Cultural experiences", "Natural beauty", "Cuisine", "Historical sites"],
                            "best_time_to_visit": "Varies by season",
                            "budget_estimate": f"Varies based on {budget_range} preferences",
                            "travel_tips": ["Research local customs", "Check visa requirements", "Book accommodations in advance"]
                        }
                
                    # Get destination image
                    destination_image = get_destination_image(destination)
                    
                    # Display destination information in a nice card layout
                    st.markdown(f"## {destination_info['destination_name']}, {destination_info['country']}")
                    
                    if destination_image:
                        st.image(destination_image, use_column_width=True, caption=f"Beautiful {destination_info['destination_name']}")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"### About")
                        st.markdown(f"*{destination_info['description']}*")
                        
                        st.markdown("### Must-See Highlights")
                        for highlight in destination_info['highlights']:
                            st.markdown(f"- {highlight}")
                    
                    with col2:
                        st.markdown("### Travel Details")
                        st.markdown(f"**Best Time to Visit:** {destination_info['best_time_to_visit']}")
                        st.markdown(f"**Budget Estimate:** {destination_info['budget_estimate']}")
                        
                        st.markdown("### Travel Tips")
                        for tip in destination_info['travel_tips']:
                            st.markdown(f"- {tip}")
                
                except Exception as e:
                    st.error(f"Error generating recommendations. Please try again.")
                    st.write(f"Error details: {str(e)}")
                
            else:
                # Show random popular destinations if no specific search
                st.subheader("Popular Destinations You Might Like")
                
                col1, col2, col3 = st.columns(3)
                
                # Display three random destinations
                random_destinations = random.sample(list(destinations.items()), 3)
                
                for (city, info), col in zip(random_destinations, [col1, col2, col3]):
                    with col:
                        st.markdown(f"### {city}, {info['country']}")
                        st.image(info["image"], use_column_width=True)
                        st.markdown(f"*{info['description']}*")
                        
                        with st.expander("See Attractions"):
                            for attraction in info["attractions"]:
                                st.markdown(f"- {attraction}")
                        
                        st.markdown(f"**Best Time to Visit:** {info['best_season']}")
                        st.markdown(f"**Daily Budget:** {format_currency(info['budget_range'][budget_range])}")

with tab2:
    st.markdown("## Plan Your Perfect Itinerary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        itinerary_destination = st.text_input("Destination", placeholder="City or Region", key="itinerary_destination")
        
        trip_duration = st.slider("Trip Duration (Days)", min_value=1, max_value=14, value=3)
        
        interests = st.multiselect(
            "Interests",
            options=["Historical Sites", "Museums", "Nature", "Food & Dining", "Shopping", 
                     "Entertainment", "Adventure Activities", "Relaxation", "Local Culture"],
            default=["Historical Sites", "Local Culture"]
        )
        
    with col2:
        start_date = st.date_input("Start Date", value=datetime.date.today() + datetime.timedelta(days=30))
        
        pace_preference = st.select_slider(
            "Pace Preference",
            options=["Relaxed", "Moderate", "Busy"],
            value="Moderate"
        )
        
        special_requirements = st.text_area("Special Requirements or Preferences", 
                                           placeholder="E.g., traveling with kids, accessibility needs, etc.")
    
    generate_button = st.button("Generate Itinerary", key="generate_button")
    
    if generate_button and itinerary_destination:
        with st.spinner("Creating your personalized itinerary..."):
            # Use Groq API to generate itinerary
            prompt = f"""
            Create a detailed day-by-day travel itinerary for a {trip_duration}-day trip to {itinerary_destination} with these preferences:
            - Start date: {start_date}
            - Interests: {', '.join(interests)}
            - Pace: {pace_preference}
            - Special requirements: {special_requirements if special_requirements else 'None'}
            
            For each day, include:
            1. A day title/theme
            2. Morning activities (with times)
            3. Lunch recommendation (with location)
            4. Afternoon activities (with times)
            5. Dinner recommendation (with location)
            6. Evening activities (if applicable)
            
            Format the response as a Python dictionary with days as keys (Day 1, Day 2, etc.) and the daily schedule as values.
            Make sure it's valid Python dictionary format that can be evaluated in Python.
            Include realistic time estimates, specific locations/venues, and brief descriptions.
            """
            
            try:
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": "You are a travel planning AI that creates detailed, realistic itineraries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    max_tokens=2000
                )
                
                result = response.choices[0].message.content
                
                # Extract the dictionary from the response
                try:
                    import re
                    # Try to extract dictionary using regex pattern
                    dict_pattern = r"\{[\s\S]*\}"
                    dict_match = re.search(dict_pattern, result)
                    if dict_match:
                        result_str = dict_match.group()
                        itinerary = eval(result_str)
                    else:
                        raise ValueError("Could not parse the itinerary")
                except:
                    # Create a simple fallback itinerary
                    itinerary = {}
                    for day in range(1, trip_duration + 1):
                        day_key = f"Day {day}"
                        current_date = start_date + timedelta(days=day-1)
                        
                        itinerary[day_key] = {
                            "date": current_date.strftime("%A, %B %d, %Y"),
                            "title": f"Exploring {itinerary_destination} - Day {day}",
                            "morning": [
                                {"time": "09:00 AM", "activity": "Breakfast at local cafe"},
                                {"time": "10:30 AM", "activity": f"Visit popular attraction in {itinerary_destination}"}
                            ],
                            "lunch": {"time": "01:00 PM", "place": "Local restaurant"},
                            "afternoon": [
                                {"time": "02:30 PM", "activity": "Continue sightseeing"},
                                {"time": "04:00 PM", "activity": "Free time for shopping or relaxation"}
                            ],
                            "dinner": {"time": "07:00 PM", "place": "Recommended restaurant"},
                            "evening": [
                                {"time": "08:30 PM", "activity": "Evening walk or leisure activity"}
                            ]
                        }
                
                # Display the itinerary
                st.markdown(f"## Your {trip_duration}-Day Itinerary for {itinerary_destination}")
                st.markdown(f"*Starting on {start_date.strftime('%A, %B %d, %Y')}*")
                
                # Create tabs for each day
                day_tabs = st.tabs([f"Day {i+1}" for i in range(trip_duration)])
                
                for i, day_tab in enumerate(day_tabs):
                    day_key = f"Day {i+1}"
                    
                    if day_key in itinerary:
                        day_info = itinerary[day_key]
                        
                        with day_tab:
                            current_date = start_date + timedelta(days=i)
                            
                            # Show day header
                            st.markdown(f"### {day_info.get('title', day_key)}")
                            st.markdown(f"*{current_date.strftime('%A, %B %d, %Y')}*")
                            
                            # Morning activities
                            st.markdown(f"""
                            <div class="day-header">
                                🌅 Morning
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if "morning" in day_info:
                                for activity in day_info["morning"]:
                                    st.markdown(f"""
                                    <div class="activity">
                                        <span class="time">{activity.get('time', '')}</span>: {activity.get('activity', '')}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Lunch
                            st.markdown(f"""
                            <div class="day-header">
                                🍽️ Lunch
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if "lunch" in day_info:
                                lunch = day_info["lunch"]
                                st.markdown(f"""
                                <div class="activity">
                                    <span class="time">{lunch.get('time', '')}</span>: {lunch.get('place', '')}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Afternoon activities
                            st.markdown(f"""
                            <div class="day-header">
                                ☀️ Afternoon
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if "afternoon" in day_info:
                                for activity in day_info["afternoon"]:
                                    st.markdown(f"""
                                    <div class="activity">
                                        <span class="time">{activity.get('time', '')}</span>: {activity.get('activity', '')}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Dinner
                            st.markdown(f"""
                            <div class="day-header">
                                🍷 Dinner
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if "dinner" in day_info:
                                dinner = day_info["dinner"]
                                st.markdown(f"""
                                <div class="activity">
                                    <span class="time">{dinner.get('time', '')}</span>: {dinner.get('place', '')}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Evening activities
                            if "evening" in day_info and day_info["evening"]:
                                st.markdown(f"""
                                <div class="day-header">
                                    🌙 Evening
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for activity in day_info["evening"]:
                                    st.markdown(f"""
                                    <div class="activity">
                                        <span class="time">{activity.get('time', '')}</span>: {activity.get('activity', '')}
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                # Add download button for itinerary (as a placeholder)
                st.download_button(
                    label="Download Itinerary as PDF",
                    data="Itinerary download placeholder",
                    file_name=f"{itinerary_destination}_itinerary.pdf",
                    mime="application/pdf",
                    disabled=True  # Currently disabled as we don't have actual PDF generation
                )
                
                # Add sharing options
                col1, col2, col3 = st.columns(3)
                col1.button("Share via Email", disabled=True)
                col2.button("Add to Calendar", disabled=True)
                col3.button("Save Itinerary", disabled=True)
                
            except Exception as e:
                st.error(f"Error generating itinerary. Please try again.")
                st.write(f"Error details: {str(e)}")

with tab3:
    st.markdown("## Travel Inspiration")
    
    # Create columns for filtering
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category = st.selectbox(
            "Category",
            ["All", "Beaches", "Mountains", "Cities", "Cultural Sites", "Wildlife", "Adventure"]
        )
    
    with col2:
        region = st.selectbox(
            "Region",
            ["Worldwide", "Europe", "Asia", "North America", "South America", "Africa", "Oceania"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Popularity", "Budget-friendly", "Luxury", "Off the beaten path"]
        )
    
    if st.button("Find Inspiration", key="inspiration_button"):
        with st.spinner("Looking for amazing destinations..."):
            # Use Groq API to generate travel inspiration
            prompt = f"""
            Generate 4 travel inspirations with these filters:
            - Category: {category if category != 'All' else 'Any'}
            - Region: {region if region != 'Worldwide' else 'Any'}
            - Sort by: {sort_by}
            
            Format the response as a Python list of dictionaries, each with:
            - name: Destination name
            - location: Country or region
            - description: 2-3 sentence captivating description
            - main_attraction: Primary highlight or reason to visit
            - best_time: Best time to visit
            - estimated_budget: Approximate daily budget in USD
            - image_keyword: One specific keyword to describe this destination for image search (e.g. "Paris Eiffel Tower night")
            
            Make sure it's valid Python format that can be evaluated.
            """
            
            try:
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": "You are a travel inspiration AI that provides diverse and interesting destination ideas."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                result = response.choices[0].message.content
                
                # Extract the list from the response
                try:
                    import re
                    # Try to extract list using regex pattern
                    list_pattern = r"\[[\s\S]*\]"
                    list_match = re.search(list_pattern, result)
                    if list_match:
                        result_str = list_match.group()
                        inspirations = eval(result_str)
                    else:
                        raise ValueError("Could not parse the inspirations")
                except:
                    # Fallback to a predefined list
                    inspirations = [
                        {
                            "name": "Santorini",
                            "location": "Greece",
                            "description": "Beautiful island with white-washed buildings and blue domes overlooking the Aegean Sea.",
                            "main_attraction": "Breathtaking sunsets and views",
                            "best_time": "April to October",
                            "estimated_budget": "$150-200 per day",
                            "image_keyword": "Santorini blue domes sunset"
                        },
                        {
                            "name": "Kyoto",
                            "location": "Japan",
                            "description": "Traditional city with over 1,600 Buddhist temples and 400 Shinto shrines.",
                            "main_attraction": "Cherry blossoms and historic temples",
                            "best_time": "March-May and October-November",
                            "estimated_budget": "$120-180 per day",
                            "image_keyword": "Kyoto temple autumn"
                        },
                        {
                            "name": "Machu Picchu",
                            "location": "Peru",
                            "description": "Ancient Incan citadel set against a backdrop of the Andes Mountains.",
                            "main_attraction": "Incredible archaeological ruins",
                            "best_time": "May to September",
                            "estimated_budget": "$100-150 per day",
                            "image_keyword": "Machu Picchu sunrise mist"
                        },
                        {
                            "name": "Serengeti",
                            "location": "Tanzania",
                            "description": "Vast ecosystem known for its annual migration of wildebeest and zebra.",
                            "main_attraction": "Safari and wildlife viewing",
                            "best_time": "June to October",
                            "estimated_budget": "$250-400 per day",
                            "image_keyword": "Serengeti wildlife migration"
                        }
                    ]
                
                # Display the inspirations in a grid
                col1, col2 = st.columns(2)
                
                for i, inspiration in enumerate(inspirations):
                    col = col1 if i % 2 == 0 else col2
                    
                    with col:
                        # Get image for the destination
                        destination_image = get_destination_image(inspiration["image_keyword"])
                        
                        st.markdown(f"""
                        <div style="background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; animation: fadeIn 0.5s ease-out forwards {i*0.2}s;">
                            <h3>{inspiration['name']}, {inspiration['location']}</h3>
                            {f'<img src="{destination_image}" style="width: 100%; border-radius: 8px; margin-bottom: 1rem;">' if destination_image else ''}
                            <p style="font-style: italic;">{inspiration['description']}</p>
                            <p><strong>Highlight:</strong> {inspiration['main_attraction']}</p>
                            <p><strong>Best Time:</strong> {inspiration['best_time']}</p>
                            <p><strong>Budget:</strong> {inspiration['estimated_budget']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error generating travel inspiration. Please try again.")
                st.write(f"Error details: {str(e)}")
    
    # Showcase section
    st.markdown("### Trending Destinations")
    
    # Display trending destinations
    trend_cols = st.columns(3)
    
    trending = [
        {"name": "Lisbon", "country": "Portugal", "tag": "Best Value European City"},
        {"name": "Bali", "country": "Indonesia", "tag": "Digital Nomad Paradise"},
        {"name": "Mexico City", "country": "Mexico", "tag": "Cultural Hotspot"}
    ]
    
    for i, trend in enumerate(trending):
        with trend_cols[i]:
            st.markdown(f"""
            <div style="background-color: white; padding: 1rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <h3>{trend['name']}, {trend['country']}</h3>
                <p style="background-color: #e0f2fe; display: inline-block; padding: 0.3rem 0.6rem; border-radius: 20px; font-size: 0.9rem;">{trend['tag']}</p>
            </div>
            """, unsafe_allow_html=True)

# Travel tip in the footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background-color: #f0f9ff; border-radius: 10px; margin-top: 2rem;">
    <h3>💡 Travel Tip of the Day</h3>
    <p>Always keep digital and physical copies of your important travel documents in separate places. This includes your passport, visa, travel insurance, and itinerary.</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding-top: 1rem;">
    <p>© 2023 TravelGenius AI. Your journey, our expertise.</p>
</div>
""", unsafe_allow_html=True)

# Load authentication config
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/airplane-mode-on.png", width=100)
    st.title("Travel Planner Pro")
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Plan Trip", "My Itineraries", "About"],
        icons=["house", "airplane", "calendar", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Authentication
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
    
    if selected == "Home":
        st.title("Welcome to Travel Planner Pro ✈️")
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2>Your AI-Powered Travel Companion</h2>
            <p>Plan your perfect trip with personalized recommendations and smart itinerary generation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Destinations", value="100+")
        with col2:
            st.metric(label="AI Recommendations", value="24/7")
        with col3:
            st.metric(label="Happy Travelers", value="10K+")
            
    elif selected == "Plan Trip":
        st.title("Plan Your Perfect Trip 🗺️")
        
        # Trip Planning Form
        with st.form("trip_planning_form"):
            col1, col2 = st.columns(2)
            with col1:
                destination = st.text_input("Destination", placeholder="e.g., Paris, France")
                start_date = st.date_input("Start Date", min_value=datetime.now().date())
                duration = st.number_input("Duration (days)", min_value=1, max_value=30, value=7)
            with col2:
                budget = st.selectbox("Budget Range", ["Budget", "Moderate", "Luxury"])
                travel_style = st.selectbox("Travel Style", ["Adventure", "Relaxation", "Culture", "Food"])
                interests = st.multiselect("Interests", ["Art", "History", "Nature", "Food", "Shopping", "Nightlife"])
            
            submitted = st.form_submit_button("Generate Itinerary")
            
            if submitted:
                with st.spinner("Generating your personalized itinerary..."):
                    # Prepare prompt for Groq
                    prompt = f"""
                    Create a detailed {duration}-day travel itinerary for {destination} with a {budget} budget.
                    Travel style: {travel_style}
                    Interests: {', '.join(interests)}
                    Start date: {start_date}
                    
                    Please provide:
                    1. Daily schedule with specific times
                    2. Recommended attractions and activities
                    3. Restaurant suggestions
                    4. Transportation tips
                    5. Budget breakdown
                    """
                    
                    # Generate itinerary using Groq
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a professional travel planner with expertise in creating detailed itineraries."},
                            {"role": "user", "content": prompt}
                        ],
                        model="mixtral-8x7b-32768",
                        temperature=0.7,
                    )
                    
                    itinerary = chat_completion.choices[0].message.content
                    
                    # Display itinerary in a beautiful format
                    st.markdown("""
                        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <h2>Your Personalized Itinerary</h2>
                            <div style='white-space: pre-wrap;'>
                                {}
                            </div>
                        </div>
                    """.format(itinerary), unsafe_allow_html=True)
                    
                    # Save itinerary
                    if 'itineraries' not in st.session_state:
                        st.session_state.itineraries = []
                    
                    st.session_state.itineraries.append({
                        'destination': destination,
                        'start_date': start_date,
                        'duration': duration,
                        'itinerary': itinerary
                    })
                    
    elif selected == "My Itineraries":
        st.title("My Saved Itineraries 📋")
        
        if 'itineraries' in st.session_state and st.session_state.itineraries:
            for idx, itinerary in enumerate(st.session_state.itineraries):
                with st.expander(f"Trip to {itinerary['destination']} ({itinerary['start_date']})"):
                    st.markdown(itinerary['itinerary'])
                    if st.button("Delete Itinerary", key=f"delete_{idx}"):
                        st.session_state.itineraries.pop(idx)
                        st.experimental_rerun()
        else:
            st.info("No saved itineraries yet. Start planning your trip!")
            
    elif selected == "About":
        st.title("About Travel Planner Pro ℹ️")
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2>Your AI-Powered Travel Companion</h2>
            <p>Travel Planner Pro is an advanced travel planning platform that uses artificial intelligence to create personalized travel itineraries. Our platform combines the power of AI with real-time travel data to provide you with the best possible travel recommendations.</p>
            
            <h3>Features</h3>
            <ul>
                <li>AI-powered itinerary generation</li>
                <li>Personalized travel recommendations</li>
                <li>Budget optimization</li>
                <li>Interactive travel planning</li>
                <li>Real-time travel data integration</li>
            </ul>
            
            <h3>Contact</h3>
            <p>For any questions or support, please contact us at support@travelplannerpro.com</p>
        </div>
        """, unsafe_allow_html=True)

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
