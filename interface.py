import pandas as pd

def clean_stock_data(input_file='AAPL_stock_data.csv', output_file='AAPL_cleaned.csv'):
    df=pd.read_csv(input_file, parse_dates=['Date'])
    if 'Adj Close' in df.columns:
        df = df.drop(columns=['Adj Close'])

    df = df.dropna()
    df = df.sort_values('Date')
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved as {output_file}")

if __name__ == '__main__':
    clean_stock_data()       
