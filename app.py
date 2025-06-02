import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import base64

# ------- ARIMA Forecast Function -------
def arima_forecast(df, order):
    df = df.copy()
    df.set_index('Date', inplace=True)
    series = df['Close']
    train = series[:-20]
    test = series[-20:]

    model = ARIMA(train, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=20)
    forecast.index = test.index

    rmse = np.sqrt(mean_squared_error(test, forecast))
    return train, test, forecast, rmse

# ------- CSV Download Helper -------
def get_table_download_link(df):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">📥 Download Forecast CSV</a>'
    return href

# ------- Streamlit UI -------
st.set_page_config(page_title="Stock Sensei", layout="wide")
st.title("📈 Stock Sensei: Predict better, Invest smarter")

uploaded_file = st.file_uploader("📤 Upload Cleaned Stock CSV", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        if 'Close' not in df.columns or 'Date' not in df.columns:
            st.error("CSV must have 'Date' and 'Close' columns.")
        else:
            st.success("✅ Data Loaded Successfully!")

            # ARIMA Params
            st.sidebar.header("⚙️ ARIMA Model Parameters")
            p = st.sidebar.slider("p (AR term)", 0, 10, 5)
            d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
            q = st.sidebar.slider("q (MA term)", 0, 10, 0)
            order = (p, d, q)

            # Forecast Button
            if st.button("🚀 Run Forecast"):
                with st.spinner("Running ARIMA forecast..."):
                    train, test, forecast, rmse = arima_forecast(df, order)

                    st.subheader("📊 Forecast Plot")
                    chart_df = pd.DataFrame({
                        "Train": train,
                        "Test": test,
                        "Forecast": forecast
                    })
                    st.line_chart(chart_df)

                    st.subheader("📉 RMSE (Root Mean Squared Error)")
                    st.metric(label="Forecast RMSE", value=f"{rmse:.4f}")

                    # Download link
                    forecast_df = pd.DataFrame({
                        "Date": forecast.index,
                        "Forecast": forecast.values
                    })
                    st.markdown(get_table_download_link(forecast_df), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
