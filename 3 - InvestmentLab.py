# Created on Mon Dec 12 00:34:13 2022
# Author: r_xngxpta; xngxpta@gmail.com
# !/usr/bin/env python310
# -*- coding: utf-8 -*-
# __author__ = 'R. Sengupta | r_xn'
# __copyright__ = 'Copyright 2023, Ledgr | www.alphaLedgr.com'
# __credits__ = ['r_xn, s.sengupta, prithvirajsengupta033@gmail.com, adasgupta@gmail.com']
# __license__ = 'Ledgr | alphaledgr.com'
# __version__ = '01.02.04'
# __maintainer__ = 'r_xn@alphaledgr.com'
# __emails__ = 'r_xn@alphaledgr.com / response@alphaledgr.com'
# __status__ = 'In active development'

# Imports
import numpy as np
import streamlit as st
import pandas as pd
import pandas_datareader as web
import plotly.express as px
import seaborn as sns
import matplotlib as plt
from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import objective_functions, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt
from pypfopt import CLA
from pypfopt import EfficientSemivariance
from pypfopt import EfficientCVaR
import copy
import os
import sys
import datetime as dt
from pypfopt.expected_returns import mean_historical_return
from pypfopt import risk_models
from pypfopt.risk_models import CovarianceShrinkage
import yfinance as yf
from ta.utils import dropna
from functools import reduce
direc = os.getcwd()
# direc = f'{direc}/Documents/Ledgr'
st.set_page_config(page_title='Ledgr | Optimization Engine', layout="wide")

logofile = f"{direc}/appdata/imgs/Ledgr_Logo_F2.png"
logofile2 = f'{direc}/appdata/imgs/Ledgr_Logo_F2.png'

# Declarations ################################################################
start_date = dt.datetime(2019, 1, 1)
end_date = dt.datetime.today()
pathtkr = f"{direc}/appdata/tickerlist_y.csv"
vidfile3 = open(f"{direc}/appdata/imgs/Optim-Anime.mp4", 'rb')
video_bytes3 = vidfile3.read()
# perf
perf_file1 = f"{direc}/appdata/session/mvar.txt"
perf_file2 = f"{direc}/appdata/session/hrp.txt"
perf_file3 = f"{direc}/appdata/session/cla.txt"
perf_file4 = f"{direc}/appdata/session/svar.txt"
perf_file5 = f"{direc}/appdata/session/cvar.txt"
tickerdb = pd.read_csv(pathtkr)
tickerlist = tickerdb["Symbol"]
logofile = f"{direc}/appdata/imgs/Ledgr_Logo_F2.png"
logofile2 = f"{direc}/appdata/imgs/Ledgr_Logo_F2.png"
st.logo(
    logofile,
    size="medium",
    link="https://alphaledgr.com/Blog/",
    icon_image=logofile,
)
# Icons and Links ###########################
ytube = f'{direc}/pages/appdata/imgs/ytube.svg'
fbook = f'{direc}/pages/appdata/imgs/fbook.svg'
insta = f'{direc}/pages/appdata/imgs/insta.svg'
linkedin = f'{direc}/pages/appdata/imgs/linkedin.svg'
ledgrblog = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F1.png'
tickerdb = pd.read_csv(pathtkr)
tickerlist = tickerdb["Symbol"]

with st.sidebar:
    st.image(logofile2, use_container_width=True)
    st.markdown("""Efficient investments are the foundations of any
                Efficient Economy.
                That said, the KPIs obtained via Technical Analyses,
                have indeed impacted living standards.
                Ledgr hones a user's Investment efficacy via tactical insights,
                leading one to various optimal scenarios (or allocations,
                                                          in other words""")
bc1, bc2 = st.columns(2)
with bc1:
    st.title(":InvestmentLab:")
    st.caption("Maximize Returns.")
    st.caption("Mitigate Risks.")
    st.info("Create Efficient Portfolios with Optimized Allocation.")
with bc2:
    st.video(video_bytes3)
# Inputs ######################################################################

with st.form("pfinputs"):
    st.subheader("Inputs for Optimization & Allocation")
    stocks_selected = st.multiselect("Select Assets for your Portfolio:  ",
                                     tickerlist)
    tav = st.number_input("Amount to be allocated:  ",
                          999, 999999999999, 100000)
    exp_returns = st.slider("What levels of Returns are expectable?",
                            0, 100, (15, 35))
    volatility_tolerance_range = st.slider("Indicate your volatility expectation",
                                           0, 30, (15, 25))
    submitted = st.form_submit_button("Submit")
    if not submitted:
        st.stop()
    if submitted:
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Thanks! Optimization en course!")
        pf_df = pd.DataFrame({"Timestamp": dt.datetime.now(),
                              "Total Allocated Amount": tav,
                              "Customers Expected Returns": exp_returns,
                              "Risk Allowance": volatility_tolerance_range})
        # st.write(pf_df)
        pf_df.to_csv(f"{direc}/appdata/PortfolioLog.csv")
        pass


@st.cache_data
def get_stock(ticker):
    ticker2 = ticker + ".NS"
    ticker2 = yf.Ticker(ticker2)
    data = ticker2.history(period='5y')
    data.to_csv(f"{direc}/appdata/OHLC/{ticker2}_data.csv", index=False)
    st.write("Data ", data.tail(4))
    try:
        data[f"{ticker}"] = data["Adj Close"]
    except Exception:
        data[f"{ticker}"] = data["Close"]
    data = data[[f"{ticker}"]]
        # st.write("Closed data last", data.tail(3))
    return data



def combine_stocks(tickers):
    data_frames = []
    for i in tickers:
        data_frames.append(get_stock(i))
        df_merged = reduce(
            lambda left, right: pd.merge(left, right, on=["Date"],
                                         how="outer"),
            data_frames)
        # st.write(df_merged.head())
        return df_merged


@st.cache_resource
def op_calc(df):
    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()
    S3 = risk_models.sample_cov(df)
    latest_prices = get_latest_prices(df)
    returns = df.pct_change()
    return mu, S, S3, latest_prices, returns


@st.cache_resource
def ef_op(mu, S):
    ef = EfficientFrontier(mu, S)
    # fig_ef = px.line(ef)
    weights_mvar = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    ef_prf = pd.DataFrame(ef.portfolio_performance(verbose=True))
    shrp = ef_prf.iloc[-1].values
    er = ef_prf.iloc[-3].values
    er = er * 100
    er = er.round(2)
    vlt = ef_prf.iloc[-2].values
    vlt = vlt * 100
    vlt = vlt.round(2)
    return ef, weights_mvar, cleaned_weights, ef_prf, shrp, er, vlt


@st.cache_resource
def hrp_op(returns):
    hrp = HRPOpt(returns)
    hrp_weights = hrp.optimize()
    hrp_prf = pd.DataFrame(hrp.portfolio_performance(verbose=True))
    shrp_hrp = hrp_prf.iloc[-1]
    er_hrp = hrp_prf.iloc[-3]
    er_hrp = er_hrp * 100
    er_hrp = er_hrp.round(2)
    vlt_hrp = ef_prf.iloc[-2].values
    vlt_hrp = vlt_hrp * 100
    vlt_hrp = vlt_hrp.round(2)
    return hrp, hrp_weights, shrp_hrp, er_hrp, vlt_hrp


@st.cache_resource
def cla_op(mu, S3):
    cla = CLA(mu, S3)
    cla_weights = cla.max_sharpe()
    cla_prf = pd.DataFrame(cla.portfolio_performance(verbose=True))
    shrp_cla = cla_prf.iloc[-1].values
    er_cla = cla_prf.iloc[-3].values
    er_cla = er_cla * 100
    er_cla = er_cla.round(2)
    vlt_cla = cla_prf.iloc[-2].values
    vlt_cla = vlt_cla * 100
    vlt_cla = vlt_cla.round(2)
    return cla, cla_weights, cla_prf, shrp_cla, er_cla, vlt_cla


@st.cache_resource
def es_op(mu, returns):
    es = EfficientSemivariance(mu, returns)
    es_weights = es.efficient_return(0.1)
    es_prf = pd.DataFrame(es.portfolio_performance())
    srtn_es = es_prf.iloc[-1]
    er_es = es_prf.iloc[-3].values
    er_es = er_es * 100
    er_es = er_es.round()
    vlt_es = es_prf.iloc[-2].values
    vlt_es = vlt_es * 100
    vlt_es = vlt_es.round(2)

    return es, es_weights, es_prf, srtn_es, er_es, vlt_es


@st.cache_resource
def ec_op(mu, returns):
    ec = EfficientCVaR(mu, returns)
    ec_weights = ec.min_cvar()
    ec_prf = pd.DataFrame(ec.portfolio_performance(verbose=True))
    # shrp_ec = ec_prf.iloc[-1].values
    er_ec = ec_prf.iloc[-2].values
    er_ec = er_ec * 100
    er_ec = er_ec.round(2)
    vlt_ec = ec_prf.iloc[-1].values
    vlt_ec = vlt_ec * 100
    vlt_ec = vlt_ec.round(2)
    return ec, ec_weights, ec_prf, er_ec, vlt_ec


st.write("----------------------------------------------------------------------------------------")
df = combine_stocks(stocks_selected)
mu, S, S3, latest_prices, returns = op_calc(df)
cx1, cx2, cx3 = st.columns(3)
with cx1:
    st.success("Calling Functions....")
    df_mu = pd.DataFrame(mu)
    df_mu.index.names = ['Securities']
    df_mu.columns.names = ['Mean Returns from Asset']

df = combine_stocks(stocks_selected)
mu, S, S3, latest_prices, returns = op_calc(df)
ef, weights_mvar, cleaned_weights, ef_prf, shrp, er, vlt = ef_op(mu, S)
hrp, hrp_weights, shrp_hrp, er_hrp, vlt_hrp = hrp_op(returns)
cla, cla_weights, cla_prf, shrp_cla, er_cla, vlt_cla = cla_op(mu, S3)
es, es_weights, es_prf, srtn_es, er_es, vlt_es = es_op(mu, returns)
ec, ec_weights, ec_prf, er_ec, vlt_ec = ec_op(mu, returns)

with cx2:
    st.success("Rendering Dashboards....")

optimizers = list(["MVAR", "HRP", "CLA", "SVAR", "CoVAR"])
returns_df = pd.DataFrame([er, er_hrp, er_cla, er_es, er_ec], index=optimizers)
returns_df.index.names = ['Optimizers']
returns_df.columns.names = ['Expected Returns Calculated']

fig_return_global = px.bar(returns_df, color=returns_df.index,
text_auto='.4s', orientation='h')
fig_return_global.update_xaxes(title='Percentage Returns Expected',
                               visible=True, showticklabels=True)
fig_return_global.update_yaxes(title='Optimal Allocations', visible=True,
                               showticklabels=True)
fig_return_global.update_layout(title="Returns Calculated",
                                title_x=0.25, showlegend=False)
# st.plotly_chart(fig_return_global)
# def figs(df, returns, mu, S):
with st.expander("Portfolio Price Data"):
    st.write(df)
fig1 = px.bar(df, text_auto='.4M')
fig1.update_xaxes(visible=False, showticklabels=False)
fig1.update_yaxes(visible=False, showticklabels=False)

fig1.update_layout(showlegend=False)
figRet = px.bar(returns, text_auto='.2M')
figRet.update_xaxes(visible=True, showticklabels=True)
figRet.update_layout(showlegend=False)
figRet.update_yaxes(title='Returns %', visible=True, showticklabels=True)
figRet.update_layout(legend=dict(
    orientation="h",
    entrywidth=60,
    yanchor="bottom",
    y=-0.14,
    xanchor="right",
    x=0.45
))
figRet.update_layout(title="Historic Returns at Daily Intervals", title_x=0.35)

figMU = px.bar(df_mu, color=df_mu.index, text_auto='.2M')
figMU.update_traces(textfont_size=12, textangle=0, textposition="outside",
                    cliponaxis=False)
figMU.update_xaxes(visible=True, showticklabels=True)
figMU.update_yaxes(title='Returns %', visible=True, showticklabels=True)
figMU.update_layout(title="Mean Returns from Assets",
                    title_x=0.25, showlegend=True)
 # figCov = px.imshow(S, title="Risk Profile [Covariance Map]")
# return fig1, figRet, figMU,  # figCov

# fig1, figRet, figMU,  # figCov = figs(df, returns, mu, S)

da_mvar = DiscreteAllocation(weights_mvar, latest_prices, tav)
allocation_mvar, leftover_mvar = da_mvar.greedy_portfolio()
names_mvar = allocation_mvar.keys()
figMVAR = px.pie(values=allocation_mvar, names=names_mvar, hole=0.5)
figMVAR.update_traces(textposition='outside', textinfo='percent+label')
# figMVAR.update_layout(height = 420)

da_hrp = DiscreteAllocation(hrp_weights, latest_prices, tav)
allocation_hrp, leftover_hrp = da_hrp.greedy_portfolio()
names_hrp = allocation_hrp.keys()
figHRP = px.pie(values=allocation_hrp, names=names_hrp, hole=0.4)
figHRP.update_traces(textposition='outside', textinfo='percent+label')
# figHRP.update_layout(height = 420)


da_cla = DiscreteAllocation(cla_weights, latest_prices, tav)
allocation_cla, leftover_cla = da_cla.greedy_portfolio()
names_cla = allocation_cla.keys()
figCLA = px.pie(values=allocation_cla, names=names_cla, hole=0.5)
figCLA.update_traces(textposition='outside', textinfo='percent+label')
# figCLA.update_layout(height = 420)

da_es = DiscreteAllocation(es_weights, latest_prices, tav)
allocation_es, leftover_es = da_es.greedy_portfolio()
names_es = allocation_es.keys()
figES = px.pie(values=allocation_es, names=names_es, hole=0.4)
figES.update_traces(textposition='outside', textinfo='percent+label')
# figES.update_layout(height = 420)

da_cvar = DiscreteAllocation(ec_weights, latest_prices, tav)
allocation_cvar, leftover_cvar = da_cvar.greedy_portfolio()
names_cvar = allocation_cvar.keys()
figCVAR = px.pie(values=allocation_cvar, names=names_cvar, hole=0.5)
figCVAR.update_traces(textposition='outside', textinfo='percent+label')
# figCVAR.update_layout(height = 420)
figMVAR.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.5,
    xanchor="center",
    x=0.5
))


figCLA.update_layout(legend=dict(
    orientation="h",
    yanchor="middle",
    y=-0.5,
    xanchor="center",
    x=0.5
))


figHRP.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.5,
    xanchor="center",
    x=0.5
))


figES.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.5,
    xanchor="center",
    x=0.5
))


figCVAR.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.23,
    xanchor="center",
    x=0.5
))
st.write("----------------------------------------------------------------------------------------")
# ################Pagework ####################################
with cx3:
    st.success("Presenting Plots and Analyses...")
x0, x1, x2, x3, x4, x5 = st.columns([2, 1, 1, 1, 1, 1])
with x0:
   st.header("Expected Returns")
with x1:
    st.metric("**Option - 1**", returns_df.iloc[-5])
    st.write("***MVAR***")
with x2:
    st.metric("**Option - 2**", returns_df.iloc[-4])
    st.write("***HRP***")
with x3:
    st.metric("**Option - 3**", returns_df.iloc[-3])
    st.write("***CLA***")
with x4:
    st.metric("**Option - 4**", returns_df.iloc[-2])
    st.write("***SVAR***")
with x5:
    st.metric("**Option - 5**", returns_df.iloc[-1])
    st.write("***coVAR***")
st.write("----------------------------------------------------------------------------------------")
with st.container():
    st.header("Trends, Returns & Risk Exposure")
    st.plotly_chart(figRet)

    x6, x8 = st.columns([2, 3])
    with x6:
        st.plotly_chart(fig_return_global)
    with x8:
        st.plotly_chart(figMU)
        # st.plotly_chart( # figCov)
st.write("----------------------------------------------------------------------------------------")
with st.container():
    cbb1, cbb2, cbb3, cbb4 = st.columns([2, 1, 1, 1])
    with cbb1:
        st.subheader('Option - 1')
        st.caption("Mean Variance Optimization")
    with cbb2:
        st.metric("Returns Expected (%)", er)
    with cbb3:
        st.metric("Volatility (%) Exposure", vlt)
    with cbb4:
        st.metric("Sharpe Ratio", shrp.round(2))
    cn1, cn2 = st.columns([1, 4])
    with cn1:
        for n in allocation_mvar:
            st.metric(f"{n}", allocation_mvar[n])
        st.info("Funds remaining: INR {:.2f}".format(leftover_mvar))
    with cn2:
        st.plotly_chart(figMVAR, use_container_width=True)
st.write("----------------------------------------------------------------------------------------")
with st.container():
    cbb4, cbb5, cbb6, cbb7 = st.columns([2, 1, 1, 1])
    with cbb4:
        st.subheader("Option - 2")
        st.caption("Hierarchical Risk Parity Algorithm")
    with cbb5:
        st.metric("Returns Expected (%)", er_hrp)
    with cbb6:
        st.metric("Volatility (%) Exposure", vlt_hrp)
    with cbb7:
        st.metric("Sharpe Ratio", shrp_hrp.round(2))
    cn3, cn4 = st.columns([1, 3])
    with cn3:
        for m in allocation_hrp:
            st.metric(f"{m}", allocation_hrp[m])
        st.info("Funds remaining (HRP): INR{:.2f}".format(leftover_hrp))
    with cn4:
        st.plotly_chart(figHRP, use_container_width=True) # )
st.write("----------------------------------------------------------------------------------------")
with st.container():
    cbb7, cbb8, cbb9, cbb10 = st.columns([2, 1, 1, 1])
    with cbb7:
        st.subheader("Option - 3")
        st.caption("Critical Line Algorithm")
    with cbb8:
        st.metric("Returns Expected (%)", er_cla)
    with cbb9:
        st.metric("Volatility (%) Exposure", vlt_cla)
    with cbb10:
        st.metric("Sharpe Ratio", shrp_cla.round(2))
    cn5, cn6 = st.columns([1, 3])
    with cn5:
        for k in allocation_cla:
            st.metric(f"{k}", allocation_cla[k])
        st.info("Funds remaining (CLA): INR {:.2f}".format(leftover_cla))
    with cn6:
        st.plotly_chart(figCLA, use_container_width=True) # )
    # st.plotly_chart(fig_cla)
st.write("----------------------------------------------------------------------------------------")
with st.container():
    cbb11, cbb12, cbb13, cbb14 = st.columns([2, 1, 1, 1])
    with cbb11:
        st.subheader("Option - 4")
        st.caption("Efficient SemiVariance Optimization")
    with cbb12:
        st.metric("Returns Expected (%)", er_es)
    with cbb13:
        st.metric("Volatility (%) Exposure", vlt_es)
    with cbb14:
        st.metric("Sortino's Ratio", srtn_es.round(2))
    cn7, cn8 = st.columns([1, 3])
    with cn7:
        for j in allocation_es:
            st.metric(f"{j}", allocation_es[j])
        st.info("Funds remaining (ES): INR {:.2f}".format(leftover_es))
    with cn8:
        st.plotly_chart(figES, use_container_width=True) #,use_container_width=True)
st.write("----------------------------------------------------------------------------------------")
with st.container():
    cbb77, cbb7a, cbb8a = st.columns(3)
    with cbb77:
        st.subheader("Option - 5")
        st.caption("Mean CoVaRiance Optimization")
    with cbb7a:
        st.metric("Annual Returns Expected (%)", er_ec)
    with cbb8a:
        st.metric("Conditional Value(%) at Risk", vlt_ec)
    cn9, cn10 = st.columns([1, 3])
    with cn9:
        for i in allocation_cvar:
            st.metric(f"{i}", allocation_cvar[i])
        st.info("Funds remaining: INR {:.2f}".format(leftover_cvar))
    with cn10:
        st.plotly_chart(figCVAR, use_container_width=True) # )
st.write("----------------------------------------------------------------------------------------")
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
    st.write(" ")
with column5:
    st.image(insta, 'Ledgr @ Instagram')

st.write("----------------------------------------------------------------------------------------")

f9, f10, f11 = st.columns([1, 5, 1])
with f9:
    st.write(" ")
with f10:
    st.caption(": | 2024 - 2025 | All Rights Resrved  ©  Ledgr Inc. | www.alphaLedgr.com | alphaLedgr Technologies Ltd. :")
with f11:
    st.write(" ")
