# Stock Price Prediction with LSTM

## Overview
This project is a web application that uses Long Short-Term Memory (LSTM) neural networks to predict stock prices. Built with Streamlit, it provides an interactive interface for users to analyze historical stock data and make predictions.

## Features
- 📈 Real-time stock data fetching using Yahoo Finance
- 🤖 LSTM-based prediction model
- 📊 Interactive visualizations with Plotly
- 📱 User-friendly interface with Streamlit
- 📁 Custom data upload capability
- 📉 Multiple performance metrics
- 🔄 Real-time model training and predictions

## How It Works
1. **Data Collection**
   - Fetches historical stock data from Yahoo Finance
   - Supports custom data upload in CSV format
   - Allows date range selection for analysis

2. **Model Architecture**
   - Uses LSTM neural networks for time series prediction
   - Implements sequence-based learning
   - Includes dropout layers to prevent overfitting

3. **Prediction Process**
   - Preprocesses data using MinMaxScaler
   - Creates sequences for LSTM input
   - Trains model on historical data
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

## Custom Data
The application supports custom data upload:
- Upload a CSV file with date and price columns
- Select the appropriate columns
- Validate the data
- Train the model on your custom dataset

## Technical Details
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: TensorFlow/Keras
- **Visualization**: Plotly
- **Data Source**: Yahoo Finance API

## Future Improvements
- [ ] Add more technical indicators
- [ ] Implement multiple model comparison
- [ ] Add portfolio optimization features
- [ ] Include sentiment analysis
- [ ] Support for multiple timeframes

## Contributing
Feel free to submit issues and enhancement requests!

## Acknowledgments
- Yahoo Finance for providing stock data
- Streamlit for the web framework
- TensorFlow/Keras for the LSTM implementation
