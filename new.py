import torch
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Set environment variables to avoid PyTorch conflicts
device = torch.device("cpu")
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["STREAMLIT_WATCHDOG"] = "false"

# Load FinBERT  for Sentiment Analysis
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Function to fetch live stock prices
def fetch_stock_price(stock_symbol, period="1d"):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period=period)
    return hist['Close'] if not hist.empty else None

# Function to fetch historical stock data with more features
def fetch_historical_stock_data(stock_symbol, period="5y"):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period=period)

    if hist.empty:
        return None

    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()  # 50-day moving average
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()  # 200-day moving average
    hist['RSI'] = compute_rsi(hist['Close'])  # Relative Strength Index (RSI)
    hist['Volatility'] = hist['Close'].pct_change().rolling(window=30).std()  # 30-day volatility
    hist.dropna(inplace=True)

    return hist

# Compute RSI (Relative Strength Index)
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fetch news articles from Yahoo Finance
def fetch_news_yfinance(stock_ticker):
    try:
        stock = yf.Ticker(stock_ticker)
        news = stock.news  # Fetch news data

        articles = []
        if news:
            for item in news[:10]:  # Get first 10 articles
                title = item.get("content", {}).get("title", "No Title")
                link = item.get("content", {}).get("canonicalUrl", {}).get("url", "No Link")
                pub_date = item.get("content", {}).get("pubDate", "Unknown Time")  # Use pubDate
                
                print(f"Title: {title}")
                print(f"Link: {link}")
                print(f"Published Time: {pub_date}")
                print("-" * 50)

                articles.append({"title": title, "link": link, "published_at": pub_date})

        if not articles:
            print("Yahoo Finance returned no news.")

        return articles

    except Exception as e:
        print(f"Error fetching Yahoo Finance news: {e}")
        return []


def analyze_sentiment(text):
    if not text:
        return "Neutral", 0

    finbert_result = finbert(text)[0]  
    finbert_label = finbert_result['label'].upper()  # Make sure it's uppercase
    if finbert_label == "POSITIVE":
        sentiment = "POSITIVE"
        score = 1
    elif finbert_label == "NEGATIVE":
        sentiment = "NEGATIVE"
        score = -1
    else:
        sentiment = "NEUTRAL"
        score = 0

    return sentiment, score


# Compute weighted sentiment score
def get_weighted_sentiment_score(stock_symbol):
    articles = fetch_news_yfinance(stock_symbol)
    sentiment_scores = []
    
    for i, article in enumerate(articles):
        sentiment, score = analyze_sentiment(article['title'])
        weight = np.exp(-0.3 * i)  # Give more weight to recent news
        sentiment_scores.append(score * weight)

    return sum(sentiment_scores)

# Fetch Nifty & Sensex Index values
def fetch_market_trends():
    nifty = fetch_stock_price("^NSEI")
    sensex = fetch_stock_price("^BSESN")
    return nifty.iloc[-1] if nifty is not None else None, sensex.iloc[-1] if sensex is not None else None

# Train XGBoost Model
def train_xgboost_model(stock_symbol):
    df = fetch_historical_stock_data(stock_symbol)
    if df is None or df.empty:
        return None

    df['Date'] = df.index.map(pd.Timestamp.toordinal)
    features = ['Date', 'SMA_50', 'SMA_200', 'RSI', 'Volatility']
    
    X = df[features]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, objective="reg:squarederror")
    model.fit(X_train, y_train)
    # Performance Analysis
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display metrics in Streamlit
    st.subheader("📊 Model Performance Metrics")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")

    return model


# Predict next stock price with sentiment adjustment
def predict_stock_price(stock_symbol):
    df = fetch_historical_stock_data(stock_symbol)
    if df is None or df.empty:
        return None

    model = train_xgboost_model(stock_symbol)
    if model is None:
        return None

    next_day = pd.Timestamp.today().toordinal() + 1
    latest_data = df.iloc[-1]
    
    next_features = np.array([[next_day, latest_data['SMA_50'], latest_data['SMA_200'], latest_data['RSI'], latest_data['Volatility']]])
    predicted_price = model.predict(next_features)[0]

    # Adjust price based on sentiment & volatility
    sentiment_score = get_weighted_sentiment_score(stock_symbol)
    volatility_factor = latest_data['Volatility']
    sentiment_impact = np.clip(sentiment_score * volatility_factor, -0.05, 0.05)  # Max ±5% impact
    adjusted_price = predicted_price * (1 + sentiment_impact)
    return adjusted_price, sentiment_score

# Calculate confidence score
def compute_confidence_score(sentiment_score, volatility):
    base_confidence = 80  # Default confidence is 80%
    sentiment_confidence = abs(sentiment_score) * 10  # Convert sentiment impact to confidence boost
    volatility_penalty = volatility * 100  # Higher volatility reduces confidence
    confidence = base_confidence + sentiment_confidence - volatility_penalty
    return max(50, min(confidence, 95))  # Confidence between 50% and 95%

# Streamlit UI
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Stock Market Prediction", page_icon="📈", layout="wide")

st.title("📊 Stock Market Prediction with Sentiment Analysis")
st.markdown("Analyze news sentiment to predict stock market trends! 🚀")

stock_symbol = st.text_input(
    "🔍 Enter Stock Symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS)")

if st.button("📌 Predict Stock Movement"):
    articles = fetch_news_yfinance(stock_symbol)
    if not articles:
        st.warning("⚠️ No news data available for sentiment analysis.")
    else:
        df = pd.DataFrame(articles)
        df["Sentiment"], df["Score"] = zip(*df["title"].apply(analyze_sentiment))
        
        st.subheader("📰 News Sentiment Analysis")
        st.dataframe(df.style.applymap(lambda x: "background-color: #ffcccc" if x == "Negative" else "background-color: #ccffcc", subset=["Sentiment"]))
        
        sentiment_chart = px.bar(df["Sentiment"].value_counts(), labels={'index': 'Sentiment', 'value': 'Count'}, title="Sentiment Distribution")
        st.plotly_chart(sentiment_chart, use_container_width=True)

    sentiment_score = get_weighted_sentiment_score(stock_symbol)
    st.metric("📊 News Sentiment Score", f"{sentiment_score:.2f}", delta_color="inverse")

    nifty, sensex = fetch_market_trends()
    if nifty and sensex:
        st.subheader("📈 Market Indices")
        col1, col2 = st.columns(2)
        col1.metric("Nifty Index", f"₹{nifty}")
        col2.metric("Sensex Index", f"₹{sensex}")

    
    predicted_price, sentiment_score = predict_stock_price(stock_symbol)

    latest_volatility = fetch_historical_stock_data(stock_symbol)['Volatility'].iloc[-1]
    confidence = compute_confidence_score(sentiment_score, latest_volatility)
    current_price = fetch_stock_price(stock_symbol)
    if current_price is not None:
        st.subheader(f"💵 Current Stock Price for {stock_symbol}")
        if stock_symbol in ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]:
            st.metric("Current Price", f"${current_price.iloc[-1]:.2f}")
        else:
            st.metric("Current Price", f"₹{current_price.iloc[-1]:.2f}")
    else:
        st.warning("⚠️ Could not fetch the current stock price.")

    if predicted_price:
        st.subheader("📉 Prediction Results")
        col1, col2 = st.columns(2)
        if stock_symbol in ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]:
            col1.metric("Predicted Stock Price", f"${predicted_price:.2f}")
        else:
            col1.metric("Predicted Stock Price", f"₹{predicted_price:.2f}")
        col2.metric("Prediction Confidence", f"{confidence}%")
    else:
        st.error("❌ Stock price prediction not available.")
    def plot_stock_prediction(stock_symbol, predicted_price):
        df = fetch_historical_stock_data(stock_symbol)
        if df is None or df.empty:
            st.error("❌ No historical data available for visualization.")
            return

        # Filter last 1 month of data
        last_month_df = df[df.index >= (df.index[-1] - pd.DateOffset(days=30))]
        #volatility = last_month_df['Volatility']
        fig = go.Figure()

        # Add actual stock price line (last 1 month)
        fig.add_trace(go.Scatter(x=last_month_df.index, y=last_month_df['Close'], mode='lines', name='Actual Price', line=dict(color='blue')))

        # Add moving averages (for last 1 month)
        fig.add_trace(go.Scatter(x=last_month_df.index, y=last_month_df['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='orange', dash='dot')))
        fig.add_trace(go.Scatter(x=last_month_df.index, y=last_month_df['SMA_200'], mode='lines', name='200-Day SMA', line=dict(color='green', dash='dot')))
        future_date = pd.Timestamp.today()
        fig.add_trace(go.Scatter(x=[future_date], y=[predicted_price], mode='markers', name='Predicted Price', marker=dict(color='red', size=10)))

        fig.update_layout(
            title=f"Last 1 Month Stock Price Trend for {stock_symbol}",
            xaxis_title="Date",
            yaxis_title="Stock Price",
            legend_title="Legend",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)


    plot_stock_prediction(stock_symbol, predicted_price)
    
        
