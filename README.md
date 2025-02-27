# AI-Driven-Stock-Price-Prediction-Using-News-Headlines-and-Market-Factors
This project leverages AI-powered sentiment analysis and XGBoost-based stock prediction to forecast stock price movements. By integrating FinBERT for financial news sentiment analysis, Yahoo Finance API for live stock data, and XGBoost regression for stock trend predictions, the model provides insightful stock forecasts with sentiment-based adjustments.

## Features\
* Live Stock Data Fetching: Uses Yahoo Finance API to retrieve stock prices and historical data.
* Sentiment Analysis with FinBERT: Analyzes financial news headlines to determine market sentiment.
* Stock Price Prediction with XGBoost: Predicts future stock prices based on historical trends, moving averages, RSI, and volatility.
* Market Index Tracking: Fetches Nifty 50 and Sensex values for market insights.
* Real-time Sentiment Scoring: Calculates a weighted sentiment score to adjust stock price predictions.
* Interactive Dashboard with Streamlit: Displays stock trends, sentiment scores, and predictions visually.

## Tech Stack
Python (Pandas, NumPy, Scikit-learn, XGBoost, Transformers)\
Streamlit (for interactive UI)\
Yahoo Finance API (for stock market data)\
Plotly & Matplotlib (for data visualization)\

## How to Run
Clone the repository:\
https://github.com/tejus2002/AI-Driven-Stock-Price-Prediction-Using-News-Headlines-and-Market-Factors.git

cd stock-market-prediction\
Install dependencies:\
pip install -r requirements.txt\
Run the Streamlit app:\
streamlit run app.py

## Future Enhancements
* Deep Learning Integration (LSTMs for advanced time-series prediction)
* Sentiment Analysis from Social Media (Twitter, Reddit)
* Portfolio Optimization Recommendations

🚀 Contributions & Feedback are welcome!
