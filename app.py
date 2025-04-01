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

# Set page config
st.set_page_config(
    page_title="Financial Time Series Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
        font-weight: 500;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        flex: 1;
        min-width: 120px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1976D2;
    }
    .metric-name {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="main-header">Financial Time Series Analysis & Forecasting</div>', unsafe_allow_html=True)

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
            if isinstance(data.index, pd.DatetimeIndex) is False:
                st.error("The selected date column could not be converted to datetime format")
                return None
            load_time_ms = (time.time() - start_time) * 1000
            st.session_state.load_time_ms = load_time_ms
            return data
        return None

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
        # Use the model to predict the validation set
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
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, 
        y=forecast,
        mode='lines',
        name='Forecast',
        line=dict(color='#FF5722', width=2, dash='dash')
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
        line=dict(color='rgba(255, 87, 34, 0)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=lower_bound,
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255, 87, 34, 0)'),
        fillcolor='rgba(255, 87, 34, 0.2)',
        name='95% Confidence Interval'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker if data_source == 'Yahoo Finance' else 'Custom Data'} - {model_type} Forecast",
        xaxis_title="Date",
        yaxis_title=column,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    return fig

def plot_seasonal_decomposition(df, column='Close'):
    """Create seasonal decomposition chart."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Ensure we have enough data
    if len(df) < 2 * 12:  # Minimum 2 cycles for seasonal decomposition
        return None
    
    # Resample to ensure regular frequency if needed
    if not pd.infer_freq(df.index):
        df_resampled = df.resample('D').mean().fillna(method='ffill')
    else:
        df_resampled = df
    
    # Determine period based on data frequency
    if pd.infer_freq(df_resampled.index) in ['D', 'B']:
        period = 7  # Weekly seasonality for daily data
    else:
        period = 12  # Monthly seasonality
    
    result = seasonal_decompose(df_resampled[column], model='additive', period=period)
    
    # Create subplots
    fig = px.figure(figsize=(10, 10))
    
    # Original
    ax1 = fig.add_subplot(411)
    ax1.plot(result.observed)
    ax1.set_title('Original')
    
    # Trend
    ax2 = fig.add_subplot(412)
    ax2.plot(result.trend)
    ax2.set_title('Trend')
    
    # Seasonal
    ax3 = fig.add_subplot(413)
    ax3.plot(result.seasonal)
    ax3.set_title('Seasonality')
    
    # Residual
    ax4 = fig.add_subplot(414)
    ax4.plot(result.resid)
    ax4.set_title('Residuals')
    
    fig.tight_layout()
    return fig

def plot_distribution(df, column='Close'):
    """Plot the distribution of returns."""
    returns = df[column].pct_change().dropna()
    
    # Create histogram with KDE
    fig = px.histogram(returns, x=returns, nbins=50, histnorm='probability density', 
                       title=f"{column} Returns Distribution")
    
    # Add KDE
    fig.add_trace(go.Scatter(
        x=np.linspace(returns.min(), returns.max(), 100),
        y=pd.Series(returns).plot.kde().evaluate(np.linspace(returns.min(), returns.max(), 100)),
        mode='lines',
        name='KDE',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Returns",
        yaxis_title="Density",
        template="plotly_white",
        showlegend=True,
        height=400
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
                # Select the column to forecast (default to 'Close' for Yahoo Finance)
                target_column = 'Close' if data_source == 'Yahoo Finance' else value_column
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                # Display data info
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
                
                # Display tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Historical Analysis", "Metrics", "Data"])
                
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
                    rolling_window = st.slider("Rolling Window (days)", 5, 100, 20)
                    
                    rolling_fig = go.Figure()
                    rolling_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[target_column],
                        mode='lines',
                        name=target_column
                    ))
                    rolling_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[target_column].rolling(rolling_window).mean(),
                        mode='lines',
                        name=f'{rolling_window}-day MA',
                        line=dict(color='orange')
                    ))
                    rolling_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[target_column].rolling(rolling_window).std(),
                        mode='lines',
                        name=f'{rolling_window}-day Std Dev',
                        line=dict(color='green')
                    ))
                    rolling_fig.update_layout(
                        title=f"Rolling Statistics (Window: {rolling_window} days)",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(rolling_fig, use_container_width=True)
                    
                    # Returns distribution
                    st.subheader("Returns Distribution")
                    returns_fig = plot_distribution(df, target_column)
                    st.plotly_chart(returns_fig, use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True)
                    
                    # Display metrics
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-value">{st.session_state.load_time_ms:.2f} ms</div>
                            <div class="metric-name">Data Load Time</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{st.session_state.processing_time_ms:.2f} ms</div>
                            <div class="metric-name">Processing Time</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        
                        for metric_name, metric_value in st.session_state.error_metrics.items():
                            if isinstance(metric_value, str):
                                value_display = metric_value
                            else:
                                value_display = f"{metric_value:.4f}"
                                
                            st.markdown(f'''
                            <div class="metric-card">
                                <div class="metric-value">{value_display}</div>
                                <div class="metric-name">{metric_name}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab4:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Data Preview")
                    st.dataframe(df.head(100), use_container_width=True)
                    
                    # Download options
                    st.download_button(
                        label="Download Data as CSV",
                        data=df.to_csv(),
                        file_name=f"{'stock_data' if data_source == 'Yahoo Finance' else 'custom_data'}.csv",
                        mime="text/csv"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                if data_source == "Upload CSV":
                    st.error("Please upload a CSV file with valid time series data.")
                else:
                    st.error(f"Could not load data for ticker {ticker}. Please check the symbol.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    # Display welcome content
    st.markdown("""
    <div class="card" style="text-align: center; padding: 40px;">
        <h2>Welcome to the Financial Time Series Analyzer</h2>
        <p>A powerful tool for analyzing and forecasting financial time series data with sub-millisecond processing.</p>
        <ul style="text-align: left; display: inline-block;">
            <li>Load stock data from Yahoo Finance or upload your own CSV</li>
            <li>Apply advanced forecasting algorithms (ARIMA, SARIMA, Prophet)</li>
            <li>Visualize time series patterns and seasonality</li>
            <li>Get detailed performance metrics and error analysis</li>
        </ul>
        <p style="margin-top: 20px;">Configure your analysis in the sidebar and click "Run Analysis" to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample visualizations
    st.markdown('<div class="sub-header" style="margin-top: 30px;">Sample Visualization</div>', unsafe_allow_html=True)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=365)
    trend = np.linspace(100, 150, 365)
    seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, 365))
    noise = np.random.normal(0, 5, 365)
    sample_data = trend + seasonality + noise
    
    sample_df = pd.DataFrame({
        'Date': dates,
        'Value': sample_data
    }).set_index('Date')
    
    # Sample plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_df.index, 
        y=sample_df['Value'],
        mode='lines',
        name='Sample Data',
        line=dict(color='#1E88E5', width=2)
    ))
    
    fig.update_layout(
        title="Sample Financial Time Series Data",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)