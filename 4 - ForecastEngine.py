# __author__ = 'R. Sengupta | r_xn'
# __copyright__ = 'Copyright 2023, Ledgr | www.alphaLedgr.com'
# __credits__ = ['r_xn, s.sengupta, adasgupta@gmail.com']
# __license__ = 'Ledgr | alphaledgr.com'
# __version__ = '01.02.04'
# __maintainer__ = 'r_xn@alphaledgr.com'
# __emails__ = 'r_xn@alphaledgr.com / outreach@alphaledgr.com'
# __status__ = 'In active development'

# Imports #####################################################################

import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
# import matplotlib as plt
# import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import os
# Page Setup ##################################################################
st.set_page_config(page_title='Ledgr | Forecasting Engine',
                   layout="wide", initial_sidebar_state="expanded")
bpath = os.getcwd()

# direc = f'{direc}/Documents/Ledgr'
logofile = f'{bpath}/pages/appdata/imgs/Ledgr_Logo_F2.png'
# logofile2 = f'{bpath}/appdata/imgs/Ledgr_Logo_F1.png'

st.logo(
    logofile,
    size="medium",
    link="https://alphaledgr.com/Blog/",
    icon_image=logofile,
)
with st.sidebar:
    st.image(logofile, use_container_width=True)
    st.caption("Select a stock, train the algorithm and predict scenarios.")
# Variables & Declarations ####################################################
start_date = dt.datetime(2020, 1, 1)
end_date = dt.datetime.today()
pathtkr = f"{bpath}/appdata/tickerlist_y.csv"
vidfile4 = open(f"{bpath}/pages/appdata/imgs/Forecast-Anime.mp4", "rb")
video_bytes4 = vidfile4.read()
tickerdb = pd.read_csv(pathtkr)
tickerlist = tickerdb["Symbol"]

# Functions & Cached Resources ################################################


@st.cache_data
def getdata(stock):
    stock = stock + ".NS"
    stock = yf.Ticker(stock)
    df = stock.history(period='max')
    return df

# Pagework 1 - Inputs #########################################################


st.title(":Forecast Engine:")
# Icons and Links ###########################
ytube = f"{bpath}/pages/appdata/imgs/ytube.svg"
fbook = f"{bpath}/pages/appdata/imgs/fbook.svg"
insta = f"{bpath}/pages/appdata/imgs/insta.svg"
linkedin = f"{bpath}/pages/appdata/imgs/linkedin.svg"
ledgrblog = f"{bpath}/pages/appdata/imgs/Ledgr_Logo_F1.png"
fc1, fc2 = st.columns([2, 3])
with fc1:
    st.caption("Train Ledgr's AI Engines. Forecast Asset Prices.")
    st.info("Chart behaviour, predict price-ranges, observe trajectories.")
with fc2:
    st.video(video_bytes4)
st.write("    -----------------------------------------------------------    ")
stock = st.selectbox("Please Select a Security Symbol", tickerlist)

df = getdata(stock)
ind = df.index
ind = ind.tz_convert(None)
open = df['Open']
hi = df['High']
lo = df['Low']
close = df['Close']
prof_df_close = pd.DataFrame({"ds": ind, "y": df['Close']})
# prof_df_close
prof_df_close = prof_df_close.reset_index()

# Pagework 2 - Forecasting  ###################################################

m = Prophet()

m.fit(prof_df_close)
future_year = m.make_future_dataframe(periods=150)
forecast_year = m.predict(future_year)
m.plot(forecast_year)
m.plot_components(forecast_year)
a = plot_plotly(m, forecast_year)
a.update_xaxes(title="Timeline", visible=True, showticklabels=True)
a.update_yaxes(title="Predicted Prices (INR)", visible=True,
               showticklabels=True)
b = plot_components_plotly(m, forecast_year)
b.update_xaxes(title="Timeline", visible=True, showticklabels=True)
b.update_yaxes(title="Predicted Prices (INR)", visible=True,
               showticklabels=True)
dx = forecast_year.filter(["ds", 'yhat'], axis=1)
dx = dx.set_index(['ds'])
dx.rename(columns={'yhat': 'Predictions'}, inplace=True)
c = px.line(dx)
c.add_trace(go.Scatter(x=dx.index, y=df['Close'], name='Close'))
c.update_xaxes(title='Timeline', showticklabels=True, visible=True)
c.update_yaxes(title="Price Data", showticklabels=True, visible=True)
c.update_layout(legend=dict(
    orientation="h",
    entrywidth=100,
    yanchor="bottom",
    y=1.02,
    xanchor="right", x=1
    ))
k1, k2, k3 = st.columns([3, 5, 3])
with k1:
    st.write(" ")
with k2:
    st.subheader("Forecast Plot")
with k3:
    st.write(" ")
st.write("  ---------------------------------------------------------------  ")
with st.container():
    st.plotly_chart(a, use_container_width=True)
    with st.expander("Get Forecast Data Here!"):
        st.write(forecast_year.iloc[-150:])
        st.write(forecast_year.iloc[-100:])
st.write("  ---------------------------------------------------------------  ")
with st.container():
    st.plotly_chart(c, use_container_width=True)
with st.container():
    j1, j2, j3 = st.columns([3, 5, 3])
    with j1:
        st.write(" ")
    with j2:
        st.subheader(f"{stock} Price Trajectory")
    with j3:
        st.write(" ")

st.plotly_chart(b, use_container_width=True)

st.write("  ---------------------------------------------------------------  ")
url = "https://www.alphaledgr.com/blog"

column1, column2, column3, column4, column5 = st.columns([1,1,1,2,1])
with column1:
    st.image(ytube, 'Ledgr\'s YouTube Channel')
with column2:
    st.image(fbook, 'Ledgr\'s FaceBook Page')
with column3:
    st.image(linkedin,  'Ledgr\'s LinkedIn Page')
with column4:
    st.write(" ")
    st.image(ledgrblog,  'Ledgr\'s own Blog', use_container_width=True)
st.write("check out this [link](%s)" % url)
st.markdown("check out this [link](%s)" % url)
with column5:
    st.image(insta, 'Check out Ledgr\'s Instagram')
#    st.image(insta, 'Ledgr @ Instagram')
ft1, ft2, ft3 = st.columns([1, 5, 1])
with ft1:
    st.write(" ")
with ft2:
    st.caption(": | 2024 - 2025 | All Rights Reserved  ©  Ledgr Inc. | www.alphaLedgr.com | alphaLedgr Technologies Ltd. :")
with ft3:
    st.write(" ")

