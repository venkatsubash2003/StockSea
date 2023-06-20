import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


start = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")
st.title("Stock prediction app")
stocks = ["AAPL", "GOOG", "MSFT", "TSLA", "IBM",
          "ORCL", "INFY", "HCLTECH.NS", "ADBE", "WIT"]
input = st.text_input("Enter a stock:")
stocks.append(input)
selected_stocks = st.selectbox("Select the Stock Code", stocks)
n_years = st.slider("years of experience:", 1, 4)
period = n_years * 365


@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, start, today)
    # data["Date"] = pd.to_datetime(data["Date"]).strftime("%Y-%m-%d")
    data.reset_index(inplace=True)

    return data


data_load_state = st.text("data loading...")
data = load_data(selected_stocks)
data_load_state.text("Loading Data... Done.")

st.subheader("Raw data")
st.write(data.tail(10))


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"],
                  name="Open data", line=dict(color='violet')))

    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["Close"], name="Close data", line=dict(color="skyblue")))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()


df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


st.subheader("Forecast data")
st.write(forecast.tail(10))


st.write("forecast data")
fig1 = plot_plotly(m, forecast)


st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

