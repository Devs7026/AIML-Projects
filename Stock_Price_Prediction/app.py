import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page config with a clean theme
st.set_page_config(
    page_title="Stock Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add minimal custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0052a3;
    }
    .metric-card {
        background-color: black;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for model caching
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Calculate technical indicators
        data['SMA_20'] = calculate_sma(data['Close'], window=20)
        data['RSI_14'] = calculate_rsi(data['Close'], window=14)
        
        # Drop NaN values
        data = data.dropna()
        
        if len(data) < 60:  # Minimum required for sequence length
            raise ValueError("Not enough data points for analysis")
            
        return data
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def calculate_sma(data, window=20):
    """Calculate Simple Moving Average."""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def preprocess_data(data, sequence_length=60, train_split=0.8):
    # Select features for scaling
    features = ['Close', 'SMA_20', 'RSI_14']
    feature_data = data[features]
    
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, :])
        y.append(scaled_data[i, 0])  # Predict Close price
    
    X, y = np.array(X), np.array(y)
    
    split_idx = int(train_split * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler, data, features

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_stock_data(data, predictions=None):
    fig = go.Figure()
    
    # Plot actual prices
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Actual Price',
        line=dict(color='#0066cc', width=2)
    ))
    
    # Plot SMA
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_20'],
        name='20-day SMA',
        line=dict(color='#ff9900', width=1)
    ))
    
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=data.index[-len(predictions):],
            y=predictions,
            name='Predicted Price',
            line=dict(color='#28a745', width=2, dash='dash')
        ))
    
    # Add RSI subplot
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI_14'],
        name='RSI (14)',
        line=dict(color='#ff4444', width=1),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Stock Price History and Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(
            title='RSI',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate various accuracy metrics for the model predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    direction_true = np.diff(y_true.flatten())
    direction_pred = np.diff(y_pred.flatten())
    directional_accuracy = np.mean((direction_true > 0) == (direction_pred > 0)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2,
        'Directional Accuracy': directional_accuracy
    }

def plot_accuracy_metrics(data, predictions, metrics):
    """Create a plot showing actual vs predicted values with accuracy metrics."""
    fig = go.Figure()
    
    # Plot actual prices
    fig.add_trace(go.Scatter(
        x=data.index[-len(predictions):],
        y=data['Close'].values[-len(predictions):],
        name='Actual Price',
        line=dict(color='blue')
    ))
    
    # Plot predictions
    fig.add_trace(go.Scatter(
        x=data.index[-len(predictions):],
        y=predictions.flatten(),
        name='Predicted Price',
        line=dict(color='red', dash='dash')
    ))
    
    # Accuracy metrics
    metrics_text = '<br>'.join([
        f'MAE: ${metrics["MAE"]:.2f}',
        f'RMSE: ${metrics["RMSE"]:.2f}',
        f'R² Score: {metrics["R2 Score"]:.3f}',
        f'Directional Accuracy: {metrics["Directional Accuracy"]:.1f}%'
    ])
    
    fig.update_layout(
        title='Model Predictions vs Actual Prices with Accuracy Metrics',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        annotations=[{
            'x': 0.02,
            'y': 0.98,
            'xref': 'paper',
            'yref': 'paper',
            'text': metrics_text,
            'showarrow': False,
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'black',
            'borderwidth': 1,
            'borderpad': 4
        }]
    )
    
    return fig

def main():
    st.title("Stock Price Prediction with LSTM")
    
    # Add image slider
    st.markdown("""
    <style>
    .stImage {
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Keep only the second image
    image = "https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg"
    
    # Display the single image
    st.image(image, 
            caption="Stock Market Analysis", 
            use_container_width=True)
    
    # Add a small description below the image
    st.markdown("""
    <div style='text-align: center; padding: 0.5rem; background-color: black; border-radius: 4px; margin: 1rem 0;'>
        <p style='color: #e9ecef;'>Explore stock market analysis through our interactive prediction tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: black; border-radius: 4px; margin-bottom: 1rem;'>
        <p>This application uses LSTM neural networks to predict stock prices. Enter a stock ticker symbol and adjust the parameters to train the model and make predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("Input Parameters")
    
    ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL").upper()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", start_date)
    with col2:
        end_date = st.date_input("End Date", end_date)
    
    # Model parameters with improved layout
    st.sidebar.subheader("Model Parameters")
    epochs = st.sidebar.slider("Number of Epochs", 10, 100, 50)
    batch_size = st.sidebar.slider("Batch Size", 16, 64, 32)
    
    # Add a new section for custom data upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("Custom Data Training")
    
    # Add file uploader
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            custom_data = pd.read_csv(uploaded_file)
            
            # Display data preview
            st.sidebar.markdown("### Data Preview")
            st.sidebar.dataframe(custom_data.head())
            
            # Add column selection
            st.sidebar.markdown("### Select Columns")
            date_col = st.sidebar.selectbox("Select Date Column", custom_data.columns)
            price_col = st.sidebar.selectbox("Select Price Column", custom_data.columns)
            
            # Add data validation
            if st.sidebar.button("Validate Data"):
                try:
                    # Convert date column to datetime
                    custom_data[date_col] = pd.to_datetime(custom_data[date_col])
                    # Sort by date
                    custom_data = custom_data.sort_values(date_col)
                    # Set date as index
                    custom_data = custom_data.set_index(date_col)
                    # Select only the price column
                    custom_data = custom_data[[price_col]]
                    
                    st.sidebar.success("Data validated successfully!")
                    
                    # Add training button for custom data
                    if st.sidebar.button("Train on Custom Data"):
                        with st.spinner("Training model on custom data..."):
                            try:
                                # Preprocess custom data
                                X_train, X_test, y_train, y_test, scaler, data, features = preprocess_data(custom_data)
                                
                                # Build and train model
                                model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
                                history = model.fit(
                                    X_train, y_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_split=0.1,
                                    verbose=0
                                )
                                
                                # Store model and scaler in session state
                                st.session_state.model = model
                                st.session_state.scaler = scaler
                                
                                # Make predictions
                                predictions = model.predict(X_test)
                                
                                # Create a dummy array with zeros for other features
                                dummy_array = np.zeros((len(predictions), len(features)))
                                dummy_array[:, 0] = predictions.flatten()  # Put predictions in first column
                                
                                # Inverse transform
                                predictions = scaler.inverse_transform(dummy_array)[:, 0]
                                
                                # Do the same for actual values
                                dummy_array = np.zeros((len(y_test), len(features)))
                                dummy_array[:, 0] = y_test.flatten()
                                y_test_actual = scaler.inverse_transform(dummy_array)[:, 0]
                                
                                # Calculate accuracy metrics
                                metrics = calculate_accuracy_metrics(y_test_actual, predictions)
                                
                                # Display results
                                st.success("Model training completed on custom data!")
                                
                                # Display metrics
                                st.subheader("Custom Data Model Accuracy Metrics")
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h4>MAE</h4>
                                    <h3>${metrics['MAE']:.2f}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h4>RMSE</h4>
                                    <h3>${metrics['RMSE']:.2f}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h4>R² Score</h4>
                                    <h3>{metrics['R2 Score']:.3f}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h4>Directional Accuracy</h4>
                                    <h3>{metrics['Directional Accuracy']:.1f}%</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Plot predictions
                                fig_predictions = plot_accuracy_metrics(custom_data, predictions, metrics)
                                st.plotly_chart(fig_predictions, use_container_width=True)
                                
                                # Make next day prediction
                                latest_sequence = custom_data[-60:].values
                                latest_sequence_scaled = scaler.transform(latest_sequence)
                                latest_sequence_reshaped = np.reshape(latest_sequence_scaled, (1, 60, 1))
                                next_day_prediction = model.predict(latest_sequence_reshaped)
                                
                                # Create dummy array for inverse transform
                                dummy_array = np.zeros((1, len(features)))
                                dummy_array[0, 0] = next_day_prediction[0, 0]
                                next_day_price = scaler.inverse_transform(dummy_array)[0, 0]
                                
                                st.subheader("Next Day Prediction")
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <h3>Predicted Price: ${next_day_price:.2f}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Error during custom data training: {str(e)}")
                                
                except Exception as e:
                    st.sidebar.error(f"Error validating data: {str(e)}")
                    
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
    
    # Create two columns for the main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px;'>
            <img src='https://img.icons8.com/color/48/000000/line-chart.png' width='30'/>
            <h2>Historical Data</h2>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Historical Data")
        try:
            data = fetch_data(ticker, start_date, end_date)
            fig = plot_stock_data(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display basic statistics with improved styling
            st.markdown("### Basic Statistics")
            stats = data.describe()
            st.dataframe(stats.style.background_gradient(cmap='Blues', axis=0))
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
    
    with col2:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px;'>
            <img src='https://img.icons8.com/color/48/000000/artificial-intelligence.png' width='30'/>
            <h2>Model Training</h2>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Model Training")
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Fetch and prepare data
                    data = fetch_data(ticker, start_date, end_date)
                    X_train, X_test, y_train, y_test, scaler, data, features = preprocess_data(data)
                    
                    # Build and train model
                    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        verbose=0
                    )
                    
                    # Store model and scaler in session state
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    
                    # Make predictions
                    predictions = model.predict(X_test)
                    
                    # Create a dummy array with zeros for other features
                    dummy_array = np.zeros((len(predictions), len(features)))
                    dummy_array[:, 0] = predictions.flatten()  # Put predictions in first column
                    
                    # Inverse transform
                    predictions = scaler.inverse_transform(dummy_array)[:, 0]
                    
                    # Do the same for actual values
                    dummy_array = np.zeros((len(y_test), len(features)))
                    dummy_array[:, 0] = y_test.flatten()
                    y_test_actual = scaler.inverse_transform(dummy_array)[:, 0]
                    
                    # Calculate accuracy metrics
                    metrics = calculate_accuracy_metrics(y_test_actual, predictions)
                    
                    # Display results with improved styling
                    st.success("Model training completed!")
                    
                    # Display metrics in a cleaner format
                    st.subheader("Model Accuracy Metrics")
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>MAE</h4>
                        <h3>${metrics['MAE']:.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>RMSE</h4>
                        <h3>${metrics['RMSE']:.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>R² Score</h4>
                        <h3>{metrics['R2 Score']:.3f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h4>Directional Accuracy</h4>
                        <h3>{metrics['Directional Accuracy']:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Plot training history with improved styling
                    fig_history = go.Figure()
                    fig_history.add_trace(go.Scatter(
                        y=history.history['loss'],
                        name='Training Loss',
                        line=dict(color='#0066cc')
                    ))
                    fig_history.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        name='Validation Loss',
                        line=dict(color='#28a745')
                    ))
                    fig_history.update_layout(
                        title='Model Training Progress',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_history, use_container_width=True)
                    
                    # Plot predictions with accuracy metrics
                    fig_predictions = plot_accuracy_metrics(data, predictions, metrics)
                    st.plotly_chart(fig_predictions, use_container_width=True)
                    
                    # Make next day prediction
                    latest_sequence = data[features].values[-60:]
                    latest_sequence_scaled = scaler.transform(latest_sequence)
                    latest_sequence_reshaped = np.reshape(latest_sequence_scaled, (1, 60, len(features)))
                    next_day_prediction = model.predict(latest_sequence_reshaped)
                    
                    # Create dummy array for inverse transform
                    dummy_array = np.zeros((1, len(features)))
                    dummy_array[0, 0] = next_day_prediction[0, 0]
                    next_day_price = scaler.inverse_transform(dummy_array)[0, 0]
                    
                    st.subheader("Next Day Prediction")
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Predicted Price: ${next_day_price:.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add interpretation of metrics with improved styling
                    st.subheader("Understanding the Metrics")
                    st.markdown("""
                    <div style='background-color: black; padding: 1.5rem; border-radius: 8px; line-height: 1.6;'>
                        <h4 style='color: #0066cc; margin-bottom: 1rem;'>Key Performance Indicators:</h4>
                        <ul style='list-style-type: none; padding-left: 0;'>
                            <li style='margin-bottom: 0.8rem;'>
                                <strong style='color: #28a745;'>MAE (Mean Absolute Error):</strong><br>
                                <span style='color: #e9ecef;'>Average absolute difference between predicted and actual prices</span>
                            </li>
                            <li style='margin-bottom: 0.8rem;'>
                                <strong style='color: #28a745;'>RMSE (Root Mean Square Error):</strong><br>
                                <span style='color: #e9ecef;'>Square root of the average squared differences</span>
                            </li>
                            <li style='margin-bottom: 0.8rem;'>
                                <strong style='color: #28a745;'>R² Score:</strong><br>
                                <span style='color: #e9ecef;'>Proportion of variance in the dependent variable predictable from the independent variable.Should be close to 1.</span>
                            </li>
                            <li style='margin-bottom: 0.8rem;'>
                                <strong style='color: #28a745;'>Directional Accuracy:</strong><br>
                                <span style='color: #e9ecef;'>Percentage of times the model correctly predicts price movement direction</span>
                            </li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        else:
            st.info("Click the 'Train Model' button to start training the LSTM model.")
    
    # Add footer with improved styling
    st.markdown("---")
    st.markdown("""
    <div style='background-color: black; padding: 1rem; border-radius: 4px;'>
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;'>
            <img src='https://img.icons8.com/color/48/000000/help.png' width='30'/>
            <h3>How to use this app:</h3>
        </div>
        <ol>
            <li>Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)</li>
            <li>Adjust the date range if needed</li>
            <li>Modify model parameters (epochs and batch size) if desired</li>
            <li>Click 'Train Model' to start the prediction process</li>
            <li>View the results, including predictions and performance metrics</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 