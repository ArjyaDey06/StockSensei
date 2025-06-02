import yfinance as yf

def fetch_data():
    ticker = 'AAPL'
    data = yf.download(ticker, start='2018-01-01', end='2024-12-31')
    data.reset_index(inplace=True)  # adds 'Date' column
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]  # keep only essential cols
    data.to_csv('AAPL_stock_data.csv', index=False)
    print("✅ Data downloaded and saved cleanly as AAPL_stock_data.csv")

if __name__ == "__main__":
    fetch_data()
