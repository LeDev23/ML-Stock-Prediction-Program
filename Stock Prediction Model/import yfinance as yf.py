import yfinance as yf
import os

# Stock symbols with IPO dates
stocks = {
    'NVDA': {'name': 'Nvidia', 'ipo': '1999-01-01'},
    'GOOGL': {'name': 'Google', 'ipo': '2004-08-19'},
    'META': {'name': 'Meta', 'ipo': '2012-05-18'},
    'NFLX': {'name': 'Netflix', 'ipo': '2002-05-23'},
    'TSLA': {'name': 'Tesla', 'ipo': '2010-06-29'},
    'AMZN': {'name': 'Amazon', 'ipo': '1997-05-15'},
    'AAPL': {'name': 'Apple', 'ipo': '1980-12-12'}
}

# Your desired output folder
output_dir = r"D:\BS Computer Science Materials\USTP\3rd Year 2nd Semester\CS324 Machine Learning\Stock Prediction Model"
os.makedirs(output_dir, exist_ok=True)

# Download and save stock data
for symbol, info in stocks.items():
    print(f"Downloading data for {info['name']} ({symbol}) from {info['ipo']}...")
    data = yf.download(symbol, start=info['ipo'])
    filename = f"{info['name']}_{symbol}.csv"
    filepath = os.path.join(output_dir, filename)
    data.to_csv(filepath)
    print(f"✅ Saved to: {filepath}")

print("\n📂 All stock data saved in your project folder!")
