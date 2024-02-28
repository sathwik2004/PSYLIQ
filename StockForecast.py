# pip install streamlit prophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'HDB','SBI','TCS','NTPC.BO')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", line=dict(color='orange')))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Change the color of the forecast plot to orange
forecast_fig = plot_plotly(m, forecast)
for trace in forecast_fig['data']:
    trace['line']['color'] = 'orange'
# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

# Create a Plotly figure for the forecast with an orange line
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="forecast", line=dict(color='orange')))

# Customize the layout of the forecast chart
fig1.update_layout(
    title_text=f'Forecast plot for {n_years} years',
    xaxis_title='Date',
    yaxis_title='Forecast',
)

st.plotly_chart(fig1)
    
st.write(f'Forecast plot for {n_years} years')
st.plotly_chart(forecast_fig)


st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
