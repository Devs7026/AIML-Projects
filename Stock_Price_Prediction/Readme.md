# Stock Price Prediction with LSTM

## Overview
This project is a web application that uses Long Short-Term Memory (LSTM) neural networks to predict stock prices. Built with Streamlit, it provides an interactive interface for users to analyze historical stock data and make predictions.

## Features
- 📈 Real-time stock data fetching using Yahoo Finance
- 🤖 LSTM-based prediction model with multiple technical indicators
- 📊 Interactive visualizations with Plotly
- 📱 User-friendly interface with Streamlit
- �� Custom data upload capability
- 📉 Multiple performance metrics
- 🔄 Real-time model training and predictions
- 📊 Advanced technical analysis tools

## Technical Indicators
The model incorporates several technical indicators to enhance prediction accuracy:

1. **Moving Averages**
   - 20-day Simple Moving Average (SMA)
   - 50-day Simple Moving Average (SMA)

2. **Relative Strength Index (RSI)**
   - 14-day RSI calculation
   - Momentum and overbought/oversold conditions

3. **MACD (Moving Average Convergence Divergence)**
   - 12-day and 26-day EMAs
   - 9-day Signal line
   - Trend direction and momentum analysis


## How It Works
1. **Data Collection**
   - Fetches historical stock data from Yahoo Finance
   - Supports custom data upload in CSV format
   - Allows date range selection for analysis
   - Calculates multiple technical indicators

2. **Model Architecture**
   - Uses LSTM neural networks for time series prediction
   - Implements sequence-based learning
   - Includes dropout layers to prevent overfitting
   - Processes multiple technical indicators as features

3. **Prediction Process**
   - Preprocesses data using MinMaxScaler
   - Creates sequences for LSTM input
   - Trains model on historical data and technical indicators
   - Makes predictions for future prices

4. **Performance Metrics**
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - R² Score
   - Directional Accuracy

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run app.py
```

## Usage
1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
2. Select your desired date range
3. Adjust model parameters if needed
4. Click "Train Model" to start the prediction process
5. View the results and performance metrics
6. Analyze technical indicators and their impact on predictions

## Custom Data
The application supports custom data upload:
- Upload a CSV file with date and price columns
- Select the appropriate columns
- Validate the data
- Train the model on your custom dataset
- Technical indicators will be automatically calculated

## Technical Details
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: TensorFlow/Keras
- **Visualization**: Plotly
- **Data Source**: Yahoo Finance API
- **Technical Analysis**: Custom implementation of multiple indicators


## Contributing
Feel free to submit issues and enhancement requests!

## Acknowledgments
- Yahoo Finance for providing stock data
- Streamlit for the web framework
- TensorFlow/Keras for the LSTM implementation
- Technical analysis community for indicator formulas

## Video Demonstration



https://github.com/user-attachments/assets/6ee01155-1195-4abd-97fc-5666fb8d5fc2



