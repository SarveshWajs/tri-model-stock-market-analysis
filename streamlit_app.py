import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load pre-trained models
rf_classifier = joblib.load('rf_model.pkl')
lin_reg_model = joblib.load('lin_reg_model.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')

# Sidebar options
st.sidebar.header("Choose Task")
task = st.sidebar.selectbox("Select Task", ("Prediction", "Classification", "Clustering"))

# Function to get stock data from Yahoo Finance
def get_stock_data(ticker):
    df = yf.download(ticker, period='5d', interval='1d')  # Get the last 5 days of data
    df.reset_index(inplace=True)
    return df

# Get stock ticker input from the user
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter the stock ticker (e.g., AAPL, META)", value="AAPL")

# Fetch latest stock data
df = get_stock_data(ticker)
st.write(f"Latest data for {ticker}:")
st.write(df)

# Preprocess stock data for model usage
def preprocess_data(df):
    df['Lag_1'] = df['Close'].shift(1)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['High'] - df['Low']
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Feature scaling
    scaler = MinMaxScaler()
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'Lag_1', 'SMA_5', 'SMA_20', 'Volatility']] = scaler.fit_transform(
        df[['Open', 'High', 'Low', 'Close', 'Volume', 'Lag_1', 'SMA_5', 'SMA_20', 'Volatility']]
    )
    
    return df

# Preprocess the data
df = preprocess_data(df)

# Task 1: Regression - Stock Price Prediction
if task == "Prediction":
    st.subheader(f"Stock Price Prediction for {ticker}")
    features = ['Open', 'High', 'Low', 'Volume', 'Lag_1', 'SMA_5', 'SMA_20', 'Volatility']
    
    # Select the latest row for prediction
    X_new = df[features].iloc[-1].values.reshape(1, -1)
    
    # Make a prediction
    predicted_price = lin_reg_model.predict(X_new)
    
    st.write(f"Predicted closing price for tomorrow: ${predicted_price[0]:.2f}")

# Task 2: Classification - Price Movement Prediction
elif task == "Classification":
    st.subheader(f"Price Movement Classification for {ticker}")
    
    # Assuming target is to classify Price Up (1) or Down (0)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Lag_1', 'SMA_5', 'SMA_20', 'Volatility']
    
    X_new = df[features].iloc[-1].values.reshape(1, -1)
    prediction = rf_classifier.predict(X_new)
    
    if prediction[0] == 1:
        st.write(f"The model predicts that the price will go **UP** tomorrow.")
    else:
        st.write(f"The model predicts that the price will go **DOWN** tomorrow.")

# Task 3: Clustering - Grouping Stock Data
elif task == "Clustering":
    st.subheader(f"Clustering Stock Data for {ticker}")
    
    # Use the features for clustering
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Lag_1', 'SMA_5', 'SMA_20', 'Volatility']
    X = df[features]
    
    # Apply clustering
    clusters = kmeans_model.predict(X)
    df['Cluster'] = clusters
    
    st.write("Clustered data:")
    st.write(df[['Date', 'Close', 'Cluster']])
    
    # Visualize clusters using scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Close'], df['Volume'], c=clusters, cmap='viridis')
    plt.xlabel('Close Price')
    plt.ylabel('Volume')
    plt.title(f'Clustering of {ticker} Stock Data')
    st.pyplot(plt)
