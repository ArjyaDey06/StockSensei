import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # Suppress ARIMA convergence warnings

def arima_forecast(data_file='AAPL_cleaned.csv'):
    try:
        df = pd.read_csv(data_file, parse_dates=['Date'])
        df.set_index('Date', inplace=True)

        # Sanity check for 'Close' column
        if 'Close' not in df.columns:
            print("❌ 'Close' column not found in the dataset.")
            return

        series = df['Close']

        if len(series) < 30:
            print("❌ Not enough data for ARIMA (need at least 30 rows).")
            return

        # Train-test split
        train = series[:-20]
        test = series[-20:]

        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=20)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Test')
        plt.plot(test.index, forecast, label='Forecast', linestyle='--')
        plt.title('ARIMA Forecast')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Optionally save forecast
        output_df = pd.DataFrame({'Date': test.index, 'Actual': test.values, 'Forecast': forecast.values})
        output_df.to_csv('AAPL_forecast.csv', index=False)
        print("✅ Forecast saved to AAPL_forecast.csv")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    arima_forecast()
