import streamlit as st
import pickle
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

# Load the trained model for the selected stock
def load_model(stock_ticker):
    filename = f"{stock_ticker}_model.pkl"  # Load model corresponding to the selected stock
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Predict next day's price for the selected stock
def predict_next_day_price(model, data, scaler):
    last_60_days = data['Close'].values[-60:].reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)
    X_predict = np.array([last_60_days_scaled])
    X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))
    predicted_price = model.predict(X_predict)
    return scaler.inverse_transform(predicted_price)[0][0]

# Function to create a pie chart showing stock performance
def create_pie_chart(data):
    # Calculate percentage change over the last 30 days
    data['Pct_Change'] = data['Close'].pct_change() * 100
    positive = len(data[data['Pct_Change'] > 0])
    negative = len(data[data['Pct_Change'] < 0])
    
    # Pie chart labels and values
    labels = ['Positive', 'Negative']
    values = [positive, negative]
    
    # Create pie chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title_text="Stock Performance Breakdown (Last 30 Days)", 
                      showlegend=True)
    return fig

# Function to create a bar chart of monthly average closing prices
def create_bar_chart(data):
    # Resample data to get monthly average closing prices
    monthly_data = data['Close'].resample('M').mean().reset_index()
    
    # Rename columns for clarity
    monthly_data.columns = ['Month', 'Avg Close Price']
    
    # Create bar chart using Plotly
    fig = px.bar(monthly_data, x='Month', y='Avg Close Price', 
                 title="Monthly Average Closing Price",
                 labels={'Month': 'Month', 'Avg Close Price': 'Average Close Price'})
    fig.update_layout(xaxis_title="Month", yaxis_title="Average Close Price", 
                      xaxis_tickformat='%b %Y')
    return fig

# Streamlit UI
st.set_page_config(page_title="Stock Price Prediction", page_icon="📊", layout="centered")

# Title and Introduction
st.title("📈 Stock Price Prediction App")
st.markdown("""
    Welcome to the **Stock Price Prediction App**!  
    Use this app to predict the next day's stock price for major Indian companies.
    Choose a stock from the dropdown and see the prediction along with its recent performance breakdown!
    """)

# Select a stock from the list
stocks = ['INFY.NS', 'TCS.NS', 'HDFCBANK.NS', 'SUNPHARMA.NS', 'ONGC.NS', 'HINDUNILVR.NS']
selected_stock = st.selectbox("Select a Stock", stocks)

# Load stock data from Yahoo Finance
data = yf.download(selected_stock, start='2010-01-01', end='2023-01-01')

# Check if the data is empty
if data.empty:
    st.error(f"⚠️ No data available for **{selected_stock}**. Please check the ticker symbol and try again.")
else:
    # Show the first few rows of data for debugging
    st.write("### Stock Data Preview", data.head())

    # Display pie chart showing stock performance
    st.plotly_chart(create_pie_chart(data))
    
    # Display bar chart showing monthly average closing prices
    st.plotly_chart(create_bar_chart(data))

    # Load the trained model for the selected stock
    model = load_model(selected_stock)

    # Make a prediction when the button is clicked
    if st.button("🔮 Predict Next Day's Price"):
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(data['Close'].values.reshape(-1, 1))
        predicted_price = predict_next_day_price(model, data, scaler)
        st.markdown(f"### Predicted price for **{selected_stock}** next day: **₹{predicted_price:.2f}**")
        
        # Display additional information or fun facts
        st.markdown(f"#### About **{selected_stock}**:")
        st.write(f"Model used: GRU")

# Add footer or notes
st.markdown("---")
st.markdown("Made with ❤️ by Team Gambler")

# Optionally, add custom CSS to enhance design
st.markdown("""
    <style>
        .css-1d391kg {font-size: 18px;}
        .css-1v0mbd9 {font-size: 16px; font-weight: bold;}
        .stButton>button {background-color: #4CAF50; color: white;}
    </style>
""", unsafe_allow_html=True)
