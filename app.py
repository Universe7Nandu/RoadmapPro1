import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set page config
st.set_page_config(
    page_title="Financial Time Series Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, colorful UI with background effects
st.markdown("""
<style>
    /* Modern Gradient Background with Animation */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(to right, #8a2387, #e94057, #f27121);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Sub Header Styling */
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        text-shadow: 0 1px 3px rgba(0,0,0,0.2);
        margin-bottom: 0.8rem;
    }
    
    /* Card Styling with Glassmorphism effect */
    .card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric Container Styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 15px;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        flex: 1;
        min-width: 140px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.03);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(to right, #396afc, #2948ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .metric-name {
        font-size: 1rem;
        color: #4a4a4a;
        font-weight: 500;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(to right, #4776E6, #8E54E9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(to right, #3a61c3, #7644c6);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        background-color: rgba(255, 255, 255, 0.7);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(30, 30, 30, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .sub-header {
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #e0e0e0;
    }
    
    /* Widget Label Styling */
    .stSlider label, .stSelectbox label, .stDateInput label {
        color: #f0f0f0 !important;
        font-weight: 500;
    }
    
    /* Table styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* AI insights card */
    .ai-insights-card {
        background: linear-gradient(135deg, rgba(98, 0, 234, 0.6), rgba(236, 64, 122, 0.6));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .ai-insights-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Custom animation for loading effect */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
</style>

<!-- Add animated particles background -->
<div id="particles-js" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;"></div>
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        particlesJS("particles-js", {
            "particles": {
                "number": {
                    "value": 80,
                    "density": {
                        "enable": true,
                        "value_area": 800
                    }
                },
                "color": {
                    "value": "#ffffff"
                },
                "shape": {
                    "type": "circle",
                    "stroke": {
                        "width": 0,
                        "color": "#000000"
                    },
                },
                "opacity": {
                    "value": 0.3,
                    "random": false,
                },
                "size": {
                    "value": 3,
                    "random": true,
                },
                "line_linked": {
                    "enable": true,
                    "distance": 150,
                    "color": "#ffffff",
                    "opacity": 0.2,
                    "width": 1
                },
                "move": {
                    "enable": true,
                    "speed": 2,
                    "direction": "none",
                    "random": false,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                }
            },
            "interactivity": {
                "detect_on": "canvas",
                "events": {
                    "onhover": {
                        "enable": true,
                        "mode": "grab"
                    },
                    "onclick": {
                        "enable": true,
                        "mode": "push"
                    },
                    "resize": true
                },
                "modes": {
                    "grab": {
                        "distance": 140,
                        "line_linked": {
                            "opacity": 0.8
                        }
                    },
                    "push": {
                        "particles_nb": 4
                    }
                }
            },
            "retina_detect": true
        });
    });
</script>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="main-header">Financial Time Series Analyzer</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)
    
    # Data selection
    data_source = st.radio("Select data source", ["Yahoo Finance", "Upload CSV"])
    
    if data_source == "Yahoo Finance":
        ticker = st.text_input("Stock Ticker Symbol", "AAPL")
        period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
        period = st.select_slider("Time Period", options=period_options, value="1y")
    else:
        uploaded_file = st.file_uploader("Upload financial data CSV", type="csv")
        date_column = st.text_input("Date column name", "Date")
        value_column = st.text_input("Value column name", "Close")
    
    # Forecasting settings
    st.markdown("---")
    st.markdown('<div class="sub-header">Forecasting</div>', unsafe_allow_html=True)
    forecast_days = st.slider("Forecast Days", 1, 365, 30)
    model_type = st.selectbox("Model Type", ["ARIMA", "SARIMA", "Prophet"])
    
    if model_type in ["ARIMA", "SARIMA"]:
        with st.expander("Model Parameters"):
            p = st.slider("p (AR order)", 0, 5, 2)
            d = st.slider("d (Differencing)", 0, 2, 1)
            q = st.slider("q (MA order)", 0, 5, 2)
            if model_type == "SARIMA":
                seasonal_p = st.slider("Seasonal P", 0, 2, 1)
                seasonal_d = st.slider("Seasonal D", 0, 1, 1)
                seasonal_q = st.slider("Seasonal Q", 0, 2, 1)
                s = st.slider("Seasonal Period", [7, 12, 24, 30, 52], 12)
    
    # Add AI Assistant Section
    st.markdown("---")
    st.markdown('<div class="sub-header">AI Assistant</div>', unsafe_allow_html=True)
    
    ai_enabled = st.checkbox("Enable Groq AI Insights", value=True)
    
    if ai_enabled:
        if not GROQ_API_KEY:
            st.warning("Groq API Key not configured. Add it to your .env file to enable AI insights.")
        else:
            st.success("Groq AI insights enabled!")
            
        ai_model = st.selectbox(
            "Select Groq Model", 
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"], 
            index=0
        )
    
    run_analysis = st.button("Run Analysis", use_container_width=True)

# Main content
def load_data():
    if data_source == "Yahoo Finance":
        start_time = time.time()
        data = yf.download(ticker, period=period)
        load_time_ms = (time.time() - start_time) * 1000
        st.session_state.load_time_ms = load_time_ms
        return data
    else:
        if uploaded_file is not None:
            start_time = time.time()
            data = pd.read_csv(uploaded_file)
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
            if not isinstance(data.index, pd.DatetimeIndex):
                st.error("The selected date column could not be converted to datetime format")
                return None
            load_time_ms = (time.time() - start_time) * 1000
            st.session_state.load_time_ms = load_time_ms
            return data
        return None

def generate_groq_insights(df, forecast, model_type):
    """Generate AI insights using Groq API"""
    if not GROQ_API_KEY or not ai_enabled:
        return "AI insights are disabled or Groq API key is not configured. Add it to your .env file."
    
    try:
        import groq
        
        # Initialize Groq client
        client = groq.Client(api_key=GROQ_API_KEY)
        
        # Extract key statistics from data
        latest_price = df['Close'].iloc[-1]
        price_change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        avg_volume = df['Volume'].mean() if 'Volume' in df.columns else "N/A"
        volatility = df['Close'].pct_change().std() * 100
        forecast_change = (forecast[-1] - latest_price) / latest_price * 100
        
        # Create prompt for the Groq API
        prompt = f"""
        As a financial analyst, provide a brief but insightful analysis based on the following data:
        
        Ticker: {ticker if data_source == 'Yahoo Finance' else 'Custom data'}
        Date Range: {df.index.min().date()} to {df.index.max().date()}
        Current Price: ${latest_price:.2f}
        Overall Price Change: {price_change:.2f}%
        Volatility (Std Dev of Returns): {volatility:.2f}%
        Average Volume: {avg_volume}
        
        Forecast Model Used: {model_type}
        Forecast Prediction ({forecast_days} days): Ending at ${forecast[-1]:.2f} ({forecast_change:.2f}%)
        
        Please provide:
        1. A concise trend analysis (2-3 sentences)
        2. Key risk factors to watch (1-2 points)
        3. A brief investment recommendation
        
        Keep the entire response under 200 words and in the same format as requested.
        """
        
        # Call Groq API
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial analyst providing concise, data-driven insights."},
                {"role": "user", "content": prompt}
            ],
            model=ai_model,
            max_tokens=400
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"AI insights error: {str(e)}"

def train_forecast_model(df, column='Close'):
    """Train a time series model and generate forecasts."""
    train_data = df[column].values
    
    start_time = time.time()
    
    if model_type == "ARIMA":
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_days)
    
    elif model_type == "SARIMA":
        model = SARIMAX(train_data, 
                         order=(p, d, q), 
                         seasonal_order=(seasonal_p, seasonal_d, seasonal_q, s))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=forecast_days)
    
    elif model_type == "Prophet":
        # Prophet requires a specific dataframe format
        prophet_data = pd.DataFrame({
            'ds': df.index,
            'y': df[column]
        })
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_data)
        
        future = model.make_future_dataframe(periods=forecast_days)
        forecast_result = model.predict(future)
        
        # Extract only the forecasted period
        forecast = forecast_result.iloc[-forecast_days:]['yhat'].values
    
    processing_time_ms = (time.time() - start_time) * 1000
    st.session_state.processing_time_ms = processing_time_ms
    
    # Create dates for the forecast period
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
    
    # Calculate error metrics on a validation set (last 20% of data)
    validation_size = int(len(train_data) * 0.2)
    if validation_size > 0:
        if model_type == "ARIMA":
            history = train_data[:-validation_size]
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit()
            validation_pred = model_fit.forecast(steps=validation_size)
        elif model_type == "SARIMA":
            history = train_data[:-validation_size]
            model = SARIMAX(history, 
                             order=(p, d, q), 
                             seasonal_order=(seasonal_p, seasonal_d, seasonal_q, s))
            model_fit = model.fit(disp=False)
            validation_pred = model_fit.forecast(steps=validation_size)
        elif model_type == "Prophet":
            prophet_data = pd.DataFrame({
                'ds': df.index,
                'y': df[column]
            })
            history = prophet_data.iloc[:-validation_size]
            model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            model.fit(history)
            future = model.make_future_dataframe(periods=validation_size)
            forecast_result = model.predict(future)
            validation_pred = forecast_result.iloc[-validation_size:]['yhat'].values
        
        validation_actual = train_data[-validation_size:]
        
        mae = mean_absolute_error(validation_actual, validation_pred)
        mse = mean_squared_error(validation_actual, validation_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((validation_actual - validation_pred) / validation_actual)) * 100
    else:
        mae, mse, rmse, mape = 0, 0, 0, 0
    
    st.session_state.error_metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': f"{mape:.2f}%"
    }
    
    return forecast_dates, forecast

def plot_data_and_forecast(df, forecast_dates, forecast, column='Close'):
    """Create interactive plots for the data and forecasts."""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[column],
        mode='lines',
        name='Historical Data',
        line=dict(color='#4361EE', width=2)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, 
        y=forecast,
        mode='lines',
        name='Forecast',
        line=dict(color='#F72585', width=2, dash='dash')
    ))
    
    # Add confidence intervals (simplified approach)
    std_dev = df[column].std()
    upper_bound = forecast + 1.96 * std_dev
    lower_bound = forecast - 1.96 * std_dev
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=upper_bound,
        fill=None,
        mode='lines',
        line=dict(color='rgba(247, 37, 133, 0)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=lower_bound,
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(247, 37, 133, 0)'),
        fillcolor='rgba(247, 37, 133, 0.2)',
        name='95% Confidence Interval'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker if data_source == 'Yahoo Finance' else 'Custom Data'} - {model_type} Forecast",
        xaxis_title="Date",
        yaxis_title=column,
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(color='white')
    )
    
    return fig

def plot_seasonal_decomposition(df, column='Close'):
    """Create seasonal decomposition chart."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Ensure we have enough data
    if len(df) < 24:
        return None
    
    # Resample if needed
    if not pd.infer_freq(df.index):
        df_resampled = df.resample('D').mean().fillna(method='ffill')
    else:
        df_resampled = df
    
    # Determine period based on frequency
    if pd.infer_freq(df_resampled.index) in ['D', 'B']:
        period = 7
    else:
        period = 12
    
    result = seasonal_decompose(df_resampled[column], model='additive', period=period)
    
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, 
                        subplot_titles=("Original", "Trend", "Seasonality", "Residuals"),
                        shared_xaxes=True)
    
    fig.add_trace(
        go.Scatter(x=result.observed.index, y=result.observed, name="Original", line=dict(color="#4CC9F0")),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=result.trend.index, y=result.trend, name="Trend", line=dict(color="#4361EE")),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=result.seasonal.index, y=result.seasonal, name="Seasonality", line=dict(color="#3A0CA3")),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=result.resid.dropna().index, y=result.resid.dropna(), name="Residuals", line=dict(color="#F72585")),
        row=4, col=1
    )
    
    fig.update_layout(
        height=800, 
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(color='white')
    )
    
    return fig

def plot_distribution(df, column='Close'):
    """Plot the distribution of returns."""
    returns = df[column].pct_change().dropna()
    
    fig = px.histogram(returns, x=returns, nbins=50, histnorm='probability density', 
                       title=f"{column} Returns Distribution")
    
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(returns)
    x_vals = np.linspace(returns.min(), returns.max(), 100)
    y_vals = kde(x_vals)
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='KDE',
        line=dict(color='#F72585', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Returns",
        yaxis_title="Density",
        template="plotly_dark",
        showlegend=True,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.05)',
        font=dict(color='white')
    )
    
    return fig

# Main app logic
if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False

if run_analysis:
    st.session_state.run_clicked = True

if st.session_state.run_clicked:
    with st.spinner("Loading data and running analysis..."):
        try:
            df = load_data()
            
            if df is not None:
                # Select the column to forecast
                target_column = 'Close' if data_source == 'Yahoo Finance' else value_column
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Points", len(df))
                with col2:
                    st.metric("Date Range", f"{df.index.min().date()} to {df.index.max().date()}")
                with col3:
                    if data_source == "Yahoo Finance":
                        latest_price = df[target_column].iloc[-1]
                        previous_price = df[target_column].iloc[-2]
                        percent_change = (latest_price - previous_price) / previous_price * 100
                        st.metric("Latest Price", f"${latest_price:.2f}", f"{percent_change:.2f}%")
                    else:
                        latest_value = df[target_column].iloc[-1]
                        st.metric("Latest Value", f"{latest_value:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Generate forecasts
                forecast_dates, forecast = train_forecast_model(df, target_column)
                
                # Display tabs for visualizations
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecast", "Historical Analysis", "Metrics", "Data", "AI Insights"])
                
                with tab1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    forecast_fig = plot_data_and_forecast(df, forecast_dates, forecast, target_column)
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="sub-header">Statistical Analysis</div>', unsafe_allow_html=True)
                    
                    # Rolling statistics
                    st.subheader("Rolling Statistics")
                    rolling_window = st.slider("Rolling Window (days)", 5, 100, 20, key="rolling_window")
                    
                    rolling_fig = go.Figure()
                    rolling_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[target_column],
                        mode='lines',
                        name=target_column,
                        line=dict(color='#4CC9F0')
                    ))
                    rolling_mean = df[target_column].rolling(window=rolling_window).mean()
                    rolling_std = df[target_column].rolling(window=rolling_window).std()
                    
                    rolling_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=rolling_mean,
                        mode='lines',
                        name='Rolling Mean',
                        line=dict(color='#F72585', width=2)
                    ))
                    rolling_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=rolling_std,
                        mode='lines',
                        name='Rolling Std',
                        line=dict(color='#4361EE', width=2)
                    ))
                    
                    rolling_fig.update_layout(
                        title='Rolling Statistics',
                        xaxis_title='Date',
                        yaxis_title=target_column,
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.05)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(rolling_fig, use_container_width=True)
                    
                    # Seasonal Decomposition
                    st.subheader("Seasonal Decomposition")
                    decomposition_fig = plot_seasonal_decomposition(df, target_column)
                    if decomposition_fig:
                        st.plotly_chart(decomposition_fig, use_container_width=True)
                    else:
                        st.write("Not enough data for seasonal decomposition.")
                    
                    # Distribution Plot
                    st.subheader("Returns Distribution")
                    distribution_fig = plot_distribution(df, target_column)
                    st.plotly_chart(distribution_fig, use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="sub-header">Error Metrics & Performance</div>', unsafe_allow_html=True)
                    metrics = st.session_state.error_metrics
                    st.write("**Mean Absolute Error (MAE):**", f"{metrics['MAE']:.4f}")
                    st.write("**Mean Squared Error (MSE):**", f"{metrics['MSE']:.4f}")
                    st.write("**Root Mean Squared Error (RMSE):**", f"{metrics['RMSE']:.4f}")
                    st.write("**Mean Absolute Percentage Error (MAPE):**", metrics['MAPE'])
                    st.markdown("---")
                    st.write("**Data Load Time:**", f"{st.session_state.load_time_ms:.2f} ms")
                    st.write("**Model Processing Time:**", f"{st.session_state.processing_time_ms:.2f} ms")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab4:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="sub-header">Data Preview</div>', unsafe_allow_html=True)
                    st.dataframe(df.head(100))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab5:
                    st.markdown('<div class="card ai-insights-card">', unsafe_allow_html=True)
                    st.markdown('<div class="ai-insights-header">AI Insights</div>', unsafe_allow_html=True)
                    if ai_enabled and GROQ_API_KEY:
                        insights = generate_groq_insights(df, forecast, model_type)
                        st.write(insights)
                    else:
                        st.write("AI insights are disabled or API key is missing.")
                    st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
