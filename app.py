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

# Set page config
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Initialize session state for model caching
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']]

def preprocess_data(data, sequence_length=60, train_split=0.8):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    split_idx = int(train_split * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

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
        line=dict(color='blue')
    ))
    
    # Plot predictions 
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=data.index[-len(predictions):],
            y=predictions,
            name='Predicted Price',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title='Stock Price History and Predictions',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
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
    
   
    st.write("""
    This application uses LSTM (Long Short-Term Memory) neural networks to predict stock prices.
    Enter a stock ticker symbol and adjust the parameters to train the model and make predictions.
    """)
    
   
    st.sidebar.header("Input Parameters")
    
   
    ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL").upper()
    
   
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", start_date)
    with col2:
        end_date = st.date_input("End Date", end_date)
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    epochs = st.sidebar.slider("Number of Epochs", 10, 100, 50)
    batch_size = st.sidebar.slider("Batch Size", 16, 64, 32)
    
    # Create two columns for the main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historical Data")
        try:
            data = fetch_data(ticker, start_date, end_date)
            fig = plot_stock_data(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display basic statistics
            st.write("### Basic Statistics")
            st.write(data.describe())
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
    
    with col2:
        st.subheader("Model Training")
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Fetch and prepare data
                    data = fetch_data(ticker, start_date, end_date)
                    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
                    
                    # Build and train model
                    model = build_lstm_model((X_train.shape[1], 1))
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
                    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Calculate accuracy metrics
                    metrics = calculate_accuracy_metrics(y_test_actual, predictions)
                    
                    # Display results
                    st.success("Model training completed!")
                    
                    # Display accuracy metrics
                    st.subheader("Model Accuracy Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"${metrics['MAE']:.2f}")
                    with col2:
                        st.metric("RMSE", f"${metrics['RMSE']:.2f}")
                    with col3:
                        st.metric("R² Score", f"{metrics['R2 Score']:.3f}")
                    with col4:
                        st.metric("Directional Accuracy", f"{metrics['Directional Accuracy']:.1f}%")
                    
                    # Plot training history
                    fig_history = go.Figure()
                    fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                    fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                    fig_history.update_layout(title='Model Training Progress', xaxis_title='Epoch', yaxis_title='Loss')
                    st.plotly_chart(fig_history, use_container_width=True)
                    
                    # Plot predictions with accuracy metrics
                    fig_predictions = plot_accuracy_metrics(data, predictions, metrics)
                    st.plotly_chart(fig_predictions, use_container_width=True)
                    
                    # Make next day prediction
                    latest_sequence = data[-60:].values
                    latest_sequence_scaled = scaler.transform(latest_sequence)
                    latest_sequence_reshaped = np.reshape(latest_sequence_scaled, (1, 60, 1))
                    next_day_prediction = model.predict(latest_sequence_reshaped)
                    next_day_price = scaler.inverse_transform(next_day_prediction)[0][0]
                    
                    st.subheader("Next Day Prediction")
                    st.metric("Predicted Price", f"${next_day_price:.2f}")
                    
                    # Add interpretation of metrics
                    st.subheader("Understanding the Metrics")
                    st.write("""
                    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices
                    - **RMSE (Root Mean Square Error)**: Square root of the average squared differences
                    - **R² Score**: Proportion of variance in the dependent variable predictable from the independent variable
                    - **Directional Accuracy**: Percentage of times the model correctly predicts price movement direction
                    """)
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        else:
            st.info("Click the 'Train Model' button to start training the LSTM model.")
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    ### How to use this app:
    1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
    2. Adjust the date range if needed
    3. Modify model parameters (epochs and batch size) if desired
    4. Click 'Train Model' to start the prediction process
    5. View the results, including predictions and performance metrics
    """)

if __name__ == "__main__":
    main() 