import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

# Load pre-trained models
rf_classifier = joblib.load('rf_model.pkl')         # Classification model (Random Forest)
lin_reg_model = joblib.load('lin_reg_model.pkl')    # Regression model (Linear Regression)
kmeans_model = joblib.load('kmeans_model.pkl')      # Clustering model (KMeans)

# Initialize the scaler
close_scaler = MinMaxScaler()

# Sidebar options
st.sidebar.header("Choose Task")
task = st.sidebar.selectbox("Select Task", ("Prediction", "Classification", "Clustering"))

# Function to get stock data for META from Yahoo Finance
def get_meta_stock_data():
    # The ticker symbol for META (Facebook) is 'META'
    df = yf.download('META', period='1mo', interval='1d')  # Get the last 50 days of data
    df.reset_index(inplace=True)
    return df

# Fetch latest META stock data
df = get_meta_stock_data()
st.write("Latest data for META (Facebook):")
st.write(df.head())

# Preprocess stock data for model usage
def preprocess_data(df):
    # Adding technical indicators and other features
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_3'] = df['Close'].shift(3)
    df['Lag_7'] = df['Close'].shift(7)
    
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI (14-day period)
    rsi_indicator = RSIIndicator(df['Close'], window=14)
    df['RSI_14'] = rsi_indicator.rsi()
    
    # Bollinger Bands (20-day period)
    bb_indicator = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low'] = bb_indicator.bollinger_lband()
    
    # VWAP calculation (Cumulative sum of volume-weighted price / cumulative volume)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # Price Up: Binary indicator for classification (1 if price is up, 0 if down)
    df['Price_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Daily return calculation
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Volatility: Difference between high and low price
    df['Volatility'] = df['High'] - df['Low']

    # Handle missing values
    df.bfill(inplace=True)

    scaled_data = close_scaler.fit_transform(df['Close'])

    # Feature scaling
    scaler = MinMaxScaler()
    # Apply Min-Max scaling to the relevant features
    num_cols = df.columns.drop(['Date', 'Price_Up'])
    df[num_cols] = scaler.fit_transform(df[num_cols])

    st.write("After Data Preprocessing")
    st.write(df.head())

    return df

# Preprocess the data
df = preprocess_data(df)

# Task 1: Regression - Stock Price Prediction
if task == "Prediction":
    st.subheader(f"Stock Price Prediction for META")
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'Lag_1', 'Lag_3', 'Volatility']
    
    # Select the latest row for prediction
    X_new = df[features].iloc[-1].values.reshape(1, -1)
    
    # Make a prediction
    predicted_price_scaled = lin_reg_model.predict(X_new)
    st.write(predicted_price_scaled)
    predicted_price = close_scaler.inverse_transform(predicted_price_scaled)
    st.write(f"Predicted closing price for tomorrow: ${predicted_price[0]:.2f}")

# Task 2: Classification - Price Movement Prediction
elif task == "Classification":
    st.subheader(f"Price Movement Classification for META")
    
    features = ['Open', 'High', 'Low', 'Close', 'Lag_7']
    
    X_new = df[features].iloc[-1].values.reshape(1, -1)
    prediction = rf_classifier.predict(X_new)
    
    if prediction[0] == 1:
        st.write(f"The model predicts that the price will go **UP** tomorrow.")
    else:
        st.write(f"The model predicts that the price will go **DOWN** tomorrow.")

# Task 3: Clustering - Grouping Stock Data
elif task == "Clustering":
    st.subheader(f"Clustering Stock Data for META")
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'Lag_1', 'Volatility']
    
    # Apply clustering
    X = df[features]
    clusters = kmeans_model.predict(X)
    df['Cluster'] = clusters
    
    st.write("Clustered data:")
    st.write(df[['Date', 'Close', 'Cluster']])
    
    # Visualize clusters using scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Close'], df['Volume'], c=clusters, cmap='viridis')
    plt.xlabel('Close Price')
    plt.ylabel('Volume')
    plt.title(f'Clustering of META Stock Data')
    st.pyplot(plt)
