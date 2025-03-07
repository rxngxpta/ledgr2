# __author__ = 'R. Sengupta | r_xn'
# __copyright__ = 'Copyright 2023, Ledgr | www.alphaLedgr.com'
# __credits__ = ['r_xn, s.sengupta, prithvirajsengupta033@gmail.com]
# __license__ = 'Ledgr | alphaledgr.com'
# __version__ = '01.02.04'
# __maintainer__ = 'r_xn@alphaledgr.com'
# __emails__ = 'r_xn@alphaledgr.com / response@alphaledgr.com'
# __status__ = 'In active development'

import streamlit as st
import pandas as pd
import datetime as dt
import plotly.express as px
import yfinance as yf
# from nsepy import get_history, get_index_pe_history
import plotly.graph_objs as go
import os
# import requests
# import pickle
# import numpy as np
from plotly.subplots import make_subplots
from mftool import Mftool
# from selectolax.parser import HTMLParser

st.set_page_config(page_title='LedgrBase | Your Asset Dossier',
                   layout="wide", initial_sidebar_state="expanded")
# ##################################################################
direc = os.getcwd()
# st.write(direc)
# direc = f'{direc}/Documents/Ledgr'
mf = Mftool()
logofile = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F2.png'
# ##################################################################
st.logo(logofile, size="medium", link='https://alphaledgr.com/',
        icon_image=logofile)

with st.sidebar:
    st.image(logofile, use_container_width=True)
    st.caption("View Markets, get info on funds & monitor your Holdings!")

start_date = dt.datetime(2021, 1, 1)
end_date = dt.datetime.today()
altstart = dt.datetime(2023, 1, 1)
indlist = pd.read_csv(f'{direc}/pages/appdata/Index_L.csv')['Symbol']
indlist = pd.Series(indlist)
list_sectoral = pd.read_csv(f'{direc}/pages/appdata/Index_L_1.csv')['ySymbol']
list_thematic = pd.read_csv(f'{direc}/pages/appdata/Index_L_2.csv')['ySymbol']
list_strat = pd.read_csv(f'{direc}/pages/appdata/Index_L_3.csv')['ySymbol']
fixed_income = pd.read_csv(f'{direc}/pages/appdata/Index_L_4.csv')['ySymbol']
list_bmkt = pd.read_csv(f'{direc}/pages/appdata/Index_L_0.csv')['ySymbol']
etflist = pd.read_csv(f'{direc}/pages/appdata/ETF_L.csv')['Symbol']
tickerl = pd.read_csv(f'{direc}/pages/appdata/tickerlist_y.csv')['Symbol']
mflist = pd.read_csv(f'{direc}/pages/appdata/mfcodes.csv')['Fund Codes']
userdf = pd.read_csv("userloggedin.csv")
# ######################################################################
username_list = userdf['UserLoggedIN']
username = username_list.iloc[-1]
usertag = userdf['Usertag']
usertag = usertag.iloc[-1]
tstamp = dt.datetime.now()
userdf2 = pd.DataFrame({'UserLoggedIN': [username],
                        'Usertag': [usertag], 'Timestamp': [tstamp]})
# userdf2 = userdf2.set_index(['Timestamp'], inplace=True)
userdf = pd.concat([userdf, userdf2])
# userdf2.to_csv('userloggedin.csv', index=False)
st.title("Your Wealth Dashboard and Global Finances")
# ####################################################
# Icons and Links ###########################
ytube = f'{direc}/pages/appdata/imgs/ytube.svg'
fbook = f'{direc}/pages/appdata/imgs/fbook.svg'
insta = f'{direc}/pages/appdata/imgs/insta.svg'
linkedin = f'{direc}/pages/appdata/imgs/linkedin.svg'
ledgrblog = f'{direc}/pages/appdata/imgs/Ledgr_Logo_F1.png'
icon_size = 100  # ####################################
video_file1 = open(f'{direc}/pages/appdata/imgs/Analytics-Anime.mp4', "rb")
video_bytes1 = video_file1.read()
hh1, hh2 = st.columns([2, 3])
with hh1:
    st.header(": LedgrBase :")
    st.subheader(f'Hi {username}!')
    st.subheader("Welcome to Ledgr!")
    st.write('Organize your asset-holdings here, track their performance!')

with hh2:
    st.video(video_bytes1, format="video/mp4")

# #####################################################
try:
    user_portfolio_data = pd.read_csv(f'{username}-portfoliodata.csv')
    with st.expander("Expand here to explore your Portfolio Details.."):
        with st.container():
            st.write("User Portfolio Data", user_portfolio_data)
    # user_portfolio_data.set_index(['Portfolio Start Date'], inplace=True)
            st.button
except Exception:
    st.warning("Please Add/Build a Portfolio to Visualize their performance!")
    pass
user_portfolio_data.set_index(['Portfolio Start Date'], inplace=True)
portfolio_count = user_portfolio_data.index.unique()
# ##############################################################################
actions = ["Build an Asset Portfolio", "Upload a Portfolio File (CSV)"]
if actions == 'Upload a Portfolio File [CSV]':
    uploaded_file = st.file_uploader('Choose a (CSV)')
    if uploaded_file is None:
        st.stop()
    else:
        dataframe = pd.read_csv(uploaded_file)
        dataframe.to_csv(f'{direc}/pages/appdata/userinfo/uploadedpf.csv')
        st.write(dataframe.tail())
    with st.container():
        dataframe = dataframe.reset_index(drop=True)
        fig_pfup = px.line(dataframe)
        fig_pfup.update_layout(xaxis_title='Datetime',
                               yaxis_title='Invested Value')
        st.header('Asset Price Movements')
        st.plotly_chart(fig_pfup)

elif actions == 'Build an Asset Portfolio':
    with st.form("my_form"):
        st.write("Inside the form")
        securitylist = st.multiselect('Select Assets to add:', tickerl)
        start_date = st.date_input('Input Portfolio Start Date:')
        held = st.text_input("List purchased units seperated by comma:")
        st.date_input('Input Portfolio End Date:', end_date)
        submitted = st.form_submit_button("Submit")
        if submitted:
            units_held = held.split([", "])
            end = dt.datetime.today()
            st.success('Processing and Plotting.....')
            units = [eval(u) for u in units_held]

            ytickers = []
            for x in securitylist:
                y = x + '.NS'
                ytickers.append(y)
            df2 = pd.DataFrame({'Assets': ytickers, 'Units': units})
            df2['Portfolio Start Date'] = start_date
            df2['Portfolio End Date'] = end
            df2['Portfolio ID'] = tstamp
            # df2['Portfolio Status'] = status
            df2.to_csv('newpf.csv', index=False)
            st.write("New Portfolio", df2)
            # on = st.toggle('Add to Holdings')
            # if on:
            #     st.write('Adding to Holdings')
            #     df4 = pd.read_csv('newpf.csv')
            #     try:
            #         df0 = pd.concat([user_portfolio_data, df4])
            #         df0.to_csv(f'{username}-portfoliodata.csv', index=False)
            #         st.write('Why hello there', df0)
            #         st.write("Base Dataframe")
            #     except Exception:
            #         df4.to_csv(f'{username}-portfoliodata.csv', index=False)
            # else:
            #     st.write('Portfolio Not Added')
            #     st.subheader("Add up your Assets or Portfolio")
            #     with st.button("Add assets in a Portfolio"):
            #         actions == 'Build an Asset Portfolio'

            @st.cache_resource
            def data_BSE():
                BSE = yf.Ticker('^BSESN')
                df_BSE = BSE.history(period='5y')
                df_BSE.to_csv(f'{direc}/pages/appdata/IndexData/df_BSE.csv')
            #    di0 = df_BSE.index
                df_BSE = pd.read_csv(f'{direc}/pages/appdata/IndexData/df_BSE.csv',
                                     header=[0])
                figOHLC_BSE = go.Figure()
                figOHLC_BSE.add_trace(go.Ohlc(x=df_BSE.index,
                                              open=df_BSE["Open"],
                                              high=df_BSE["High"],
                                              low=df_BSE["Low"],
                                              close=df_BSE["Close"],
                                              name="SENSEX"))
                figOHLC_BSE.update_layout(xaxis_rangeslider_visible=False)
                return df_BSE, figOHLC_BSE

            df_BSE, figOHLC_BSE = data_BSE()


            @st.cache_resource
            def data_NSEI():
                nse = yf.Ticker('^NSEI')
                df_NSEI = nse.history(period='5y')
                df_NSEI.to_csv(f'{direc}/pages/appdata/IndexData/df_NSEI.csv')
            #    di = df_NSEI.index
                df_NSEI = pd.read_csv(f'{direc}/pages/appdata/IndexData/df_NSEI.csv',
                                      header=[0])
                figOHLC_NSEI = go.Figure()
                figOHLC_NSEI.add_trace(go.Ohlc(x=df_NSEI.index,
                                               open=df_NSEI["Open"],
                                               high=df_NSEI["High"],
                                               low=df_NSEI["Low"],
                                               close=df_NSEI["Close"],
                                               name="NIFTY50"))
                figOHLC_NSEI.update_layout(xaxis_rangeslider_visible=False)
                return df_NSEI, figOHLC_NSEI
            
            
            df_NSEI, figOHLC_NSEI = data_NSEI()
            
            
            @st.cache_resource
            def data_SPX():
                spx = yf.Ticker('^GSPC')
                df_SPX = spx.history(period='5y')
                df_SPX.to_csv(f'{direc}/pages/appdata/IndexData/df_SPX.csv')
            #    di2 = df_SPX.index
                df_SPX = pd.read_csv(f'{direc}/pages/appdata/IndexData/df_SPX.csv',
                                     header=[0])
                figOHLC_SPX = go.Figure()
                figOHLC_SPX.add_trace(go.Ohlc(x=df_SPX.index, open=df_SPX["Open"],
                                              high=df_SPX["High"],
                                              low=df_SPX["Low"], close=df_SPX["Close"],
                                              name="SPX"))
                figOHLC_SPX.update_layout(xaxis_rangeslider_visible=False)
                return df_SPX, figOHLC_SPX
            
            
            df_SPX, figOHLC_SPX = data_SPX()
            
            
            @st.cache_resource
            def data_DAX():
                dax = yf.Ticker('^GDAXI')
                df_DAX = dax.history(period='5y')
                df_DAX.to_csv(f'{direc}/pages/appdata/IndexData/df_DAX.csv')
            #    di3 = df_DAX.index
                df_DAX = pd.read_csv(f'{direc}/pages/appdata/IndexData/df_DAX.csv',
                                     header=[0])
                figOHLC_DAX = go.Figure()
                figOHLC_DAX.add_trace(go.Ohlc(x=df_DAX.index, open=df_DAX["Open"],
                                              high=df_DAX["High"],
                                              low=df_DAX["Low"],
                                              close=df_DAX["Close"],
                                              name="DAX"))
                figOHLC_DAX.update_layout(xaxis_rangeslider_visible=False)
                return df_DAX, figOHLC_DAX
            
            
            df_DAX, figOHLC_DAX = data_DAX()
            
            
            @st.cache_resource
            def data_CAC():
                cac = yf.Ticker("^FCHI")
                df_CAC = cac.history(period='5y')
                df_CAC.to_csv(f'{direc}/pages/appdata/IndexData/df_cac.csv')
            #    di4 = df_CAC.index
                df_CAC = pd.read_csv(f'{direc}/pages/appdata/IndexData/df_cac.csv',
                                     header=[0])
                figOHLC_CAC = go.Figure()
                figOHLC_CAC.add_trace(go.Ohlc(x=df_CAC.index,
                                              open=df_CAC["Open"],
                                              high=df_CAC["High"],
                                              low=df_CAC["Low"],
                                              close=df_CAC["Close"], name="CAC40"))
                figOHLC_CAC.update_layout(xaxis_rangeslider_visible=False)
                return df_CAC, figOHLC_CAC
            
            
            df_CAC, figOHLC_CAC = data_CAC()
            
            
            @st.cache_resource
            def data_DJIA():
                dji = yf.Ticker("^DJI")
                df_DJIA = dji.history(period='5y')
                df_DJIA.to_csv(f'{direc}/pages/appdata/IndexData/df_DJIA.csv')
            #    di4 = df_DJIA.index
                df_DJIA = pd.read_csv(f'{direc}/pages/appdata/IndexData/df_DJIA.csv',
                                      header=[0])
                figOHLC_DJIA = go.Figure()
                figOHLC_DJIA.add_trace(go.Ohlc(x=df_DJIA.index,
                                               open=df_DJIA["Open"], high=df_DJIA["High"],
                                               low=df_DJIA["Low"],
                                               close=df_DJIA["Close"],
                                               name="DJIA"))
                figOHLC_DJIA.update_layout(xaxis_rangeslider_visible=False)
                return df_DJIA, figOHLC_DJIA
            
            
            df_DJIA, figOHLC_DJIA = data_DJIA()
            
            
            @st.cache_resource
            def data_TYO():
                tyo = yf.Ticker("^N225")
                df_tyo = tyo.history(period='5y')
                df_tyo.to_csv(f'{direc}/pages/appdata/IndexData/df_tyo.csv')
            #    di6 = df_tyo.index
                df_tyo = pd.read_csv(f'{direc}/pages/appdata/IndexData/df_tyo.csv',
                                     header=[0])
                figOHLC_tyo = go.Figure()
                figOHLC_tyo.add_trace(go.Ohlc(x=df_tyo.index,
                                              open=df_tyo["Open"], high=df_tyo["High"],
                                              low=df_tyo["Low"], close=df_tyo["Close"],
                                              name="TYO"))
                figOHLC_tyo.update_layout(xaxis_rangeslider_visible=False)
                return df_tyo, figOHLC_tyo
            
            
            df_tyo, figOHLC_tyo = data_TYO()
            
            
            @st.cache_resource
            def data_FTSE():
                FTSE = yf.Ticker("^FTSE")
                df_FTSE = FTSE.history(period='5y')
                df_FTSE.to_csv(f'{direc}/pages/appdata/IndexData/df_FTSE.csv')
                figOHLC_FTSE = go.Figure()
            #    di7 = FTSE.index
                df_FTSE = pd.read_csv(f'{direc}/pages/appdata/IndexData/df_FTSE.csv',
                                      header=[0])
                figOHLC_FTSE.add_trace(go.Ohlc(x=df_FTSE.index,
                                               open=df_FTSE["Open"],
                                               high=df_FTSE["High"],
                                               low=df_FTSE["Low"],
                                               close=df_FTSE["Close"],
                                               name="FTSE"))
                figOHLC_FTSE.update_layout(xaxis_rangeslider_visible=False)
                return df_FTSE, figOHLC_FTSE
            
            
            df_FTSE, figOHLC_FTSE = data_FTSE()
            
            
            @st.cache_resource
            def data_mkt():
                df_mk = pd.DataFrame()
                df_mk['SENSEX'] = df_BSE['Close']
                df_mk['NSEI'] = df_NSEI['Close']
                df_mk['DAX'] = df_DAX['Close']
                df_mk['CAC'] = df_CAC['Close']
                df_mk['SPX'] = df_SPX['Close']
                df_mk['FTSE'] = df_FTSE['Close']
                df_mk['N225'] = df_tyo['Close']
                fig_mkt = go.Figure()
                fig_mkt.add_trace(go.Scatter(x=df_NSEI.index, y=df_NSEI['Close'],
                                             mode='lines', name='NSEI'))
                fig_mkt.add_trace(go.Scatter(x=df_BSE.index, y=df_BSE['Close'],
                                             mode='lines', name='SENSEX'))
                fig_mkt.add_trace(go.Scatter(x=df_DAX.index, y=df_DAX['Close'],
                                             mode='lines', name='DAX'))
                fig_mkt.add_trace(go.Scatter(x=df_CAC.index, y=df_CAC['Close'],
                                             mode='lines', name='CAC'))
                fig_mkt.add_trace(go.Scatter(x=df_SPX.index, y=df_SPX['Close'],
                                             mode='lines', name='SPX'))
                fig_mkt.add_trace(go.Scatter(x=df_DJIA.index, y=df_DJIA['Close'],
                                             mode='lines', name='DJIA'))
                fig_mkt.add_trace(go.Scatter(x=df_FTSE.index, y=df_FTSE['Close'],
                                             mode='lines', name='FTSE'))
                fig_mkt.add_trace(go.Scatter(x=df_tyo.index, y=df_tyo['Close'],
                                             mode='lines', name='N225'))
                fig_mkt.update_xaxes(visible=True, showticklabels=True)
                fig_mkt.update_yaxes(visible=True, showticklabels=True)
            
                return df_mk, fig_mkt
            
            
            df_mk, fig_mkt = data_mkt()
            multi_symbols = ['^IXIC',
                             '^GSPC', '^NYA',
                             '^BSESN',
                             '^NSEI', '^NSEBANK']
            multi_details = ['NASDAQ Composite', 'S&P500', 'NYSE Composite (DJ)',
                             'BSE SENSEX',  'NIFTY50',
                             'NIFTYBANK']
            
            multi_index_list = pd.DataFrame({"Symbol": multi_symbols,
                                             "Exchange Index": multi_details})
            
            # ^TYX - US Treasury Yield - 30 years
            
            
            @st.cache_resource
            def treasury():
                trs = yf.Ticker('^TYX')
                df_treasury = trs.history(period='5y')
                fig_treasury = go.Figure()
                fig_treasury.add_trace(go.Ohlc(x=df_treasury.index,
                                               open=df_treasury["Open"],
                                               high=df_treasury["High"],
                                               low=df_treasury["Low"],
                                               close=df_treasury["Close"]))
                fig_treasury.update_xaxes(visible=True, showticklabels=True)
                fig_treasury.update_yaxes(title='US Treasury Yield', visible=True,
                                          showticklabels=True)
                fig_treasury.update_layout(xaxis_rangeslider_visible=False,
                                           showlegend=False)
                return df_treasury, fig_treasury
            
            
            df_treasury, fig_treasury = treasury()
            
            
            @st.cache_resource
            def vix():
                vix = yf.Ticker('^VIX')
                df_vix = vix.history(period='5y')
                # df_vix = df_vix.drop(['Volume'], axis=1)
                fig_vix = go.Figure()
                fig_vix.add_trace(go.Ohlc(x=df_vix.index, open=df_vix["Open"],
                                          high=df_vix["High"],
                                          low=df_vix["Low"], close=df_vix["Close"],
                                          name="VIX"))
                fig_vix.update_traces(increasing_line_color='yellow',
                                      decreasing_line_color='gray')
                fig_vix.update_layout(xaxis_rangeslider_visible=False)
                fig_vix.update_xaxes(visible=True, showticklabels=True)
                fig_vix.update_yaxes(title='VIX', visible=True, showticklabels=True)
                fig_vix.update_layout(height=360, showlegend=False)
                return df_vix, fig_vix
            
            
            df_vix, fig_vix = vix()
            
            st.write("    ----    ")
            hg1, hg2 = st.columns([2, 3])
            with hg1:
                st.header(': MarketBoard :')
                st.subheader('Follow, Track and Global Markets')
                st.caption('Explore Indices, Exchange Traded & Mutual Funds and more!')
            with hg2:
                st.image(f'{direc}/pages/appdata/imgs/Marketbase-Anim.gif',
                         use_container_width=True)
            
            st.subheader('A. Markets & Exchanges')
            
            tub0, tub1, tub2, tub3, tub4, tub5, tub6, tub7 = st.tabs(['Global Markets',
                                                                      "NSE - IN",
                                                                      "SPX - USA",
                                                                      "DAX - GDR",
                                                                      'CAC40 - FR',
                                                                      'Dow Jones - US',
                                                                      'Nikkei225 - JPN',
                                                                      "FTSE - UK"])
            with tub0:
                st.plotly_chart(fig_mkt, use_container_width=True)
            with tub1:
                st.plotly_chart(figOHLC_NSEI, use_container_width=True)
            with tub2:
                df_SPX, figOHLC_SPX = data_SPX()
                st.plotly_chart(figOHLC_SPX, use_container_width=True)
            with tub3:
                st.plotly_chart(figOHLC_DAX, use_container_width=True)
            with tub4:
                st.plotly_chart(figOHLC_CAC, use_container_width=True)
            with tub5:
                st.plotly_chart(figOHLC_DJIA, use_container_width=True)
            with tub6:
                st.plotly_chart(figOHLC_tyo, use_container_width=True)
            with tub7:
                st.plotly_chart(figOHLC_FTSE, use_container_width=True)
            
            
            st.write("  --------  ")
            
            
            st.subheader("B. SIP Calculator")
            st.caption("Find out your Returns from any SIP scheme.")
            with st.form('sipcalc'):
                A = st.slider("Enter the monthly SIP amount: ", min_value=500,
                              max_value=9900, value=1050, step=100,
                              help="Input your monthly payments installments here!")
                YR = st.slider("Enter the yearly Rate of Return in pct: ", min_value=5,
                               max_value=20, value=10, step=1,
                               help="Indicate your scheme's Return Rate[ref: IRR or XIRR]")
                Y = st.slider("Enter the number of years: ", min_value=2,
                              max_value=15, value=5, step=1,
                              help="Indicate the number of years of investing")
                submitted = st.form_submit_button("Calculate Returns >> ")
                if submitted:
                    MR = YR/12/100
                    M = Y * 12
                    FV = A * ((((1 + MR)**(M))-1) * (1 + MR))/MR
                    FV = round(FV)
                    gh2, gh3 = st.columns(2)
                    with gh2:
                        st.subheader("Your Expected Returns are: - ")
                    with gh3:
                        st.metric("Returns [INR]", FV)
                else:
                    st.warning("Select values and click Calculate Returns")
            st.write("    ------    ")
            st.subheader('C. Mutual Funds')
            scheme_codes_mf = mf.get_scheme_codes()
            scheme_codes_mf
            all_scheme_codes = pd.DataFrame(scheme_codes_mf.items())
            mf_sel = st.selectbox("Choose Scheme Code", all_scheme_codes)
            if mf_sel == 'Select Code':
                st.warning("Choose a Mutual Fund by its Code!!")
            else:
                with st.expander("MF Details"):
                    q = mf.get_scheme_quote(mf_sel)
                    df_mf = mf.get_scheme_historical_nav(mf_sel, as_Dataframe=True)
                    df_mf = df_mf.iloc[1:]
                    #    st.write('Quotation', q)
                    scheme_det = mf.get_scheme_details(mf_sel)
                    scheme_details = pd.DataFrame(scheme_det.items(), columns=['Items',
                                                                               'Details'])
                    scheme_details = scheme_details.set_index('Items')
                    st.write('1. Fund_house', scheme_details.loc['fund_house'])
                    st.write('2. Scheme Type', scheme_details.loc['scheme_type'])
                    st.write('3. Scheme Category', scheme_details.loc['scheme_category'])
                    st.write('4. Scheme Name', scheme_details.loc['scheme_name'])
                    st.write('5. Scheme Code', scheme_details.loc['scheme_code'])
            
                    st.write(df_mf.T)
                    df_mf.reset_index()
                    # st.write(df_mf.loc[['nav']])
                    # st.bar_chart(df_mf, x=df_mf.index['nav'])
            #    st.write(df_mf.shape)
            #    f_nav = px.bar(df_mf, [["date"], ["nav"], ["dayChange"]], text_auto=True)
            #    st.plotly_chart(f_nav)
            
            st.write("   ----   ")
            
            
            @st.cache_resource
            def etf(etfselect):
                etfselect = etfselect + '.NS'
                etf = yf.Ticker(etfselect)
                df_etf = etf.history(period='5y')
                figOHLC_etf = make_subplots(rows=2, cols=1,
                                            shared_xaxes=True,
                                            vertical_spacing=0.2,
                                            subplot_titles=('NAV Price Movement',
                                                            'Traded Volume'),
                                            row_width=[0.2, 0.7])
                figOHLC_etf.add_trace(go.Ohlc(x=df_etf.index,
                                              open=df_etf["Open"],
                                              high=df_etf["High"],
                                              low=df_etf["Low"],
                                              close=df_etf["Close"],
                                              name=f"OHLC for {etfselect}"),
                                      row=1, col=1)
                figOHLC_etf.add_trace(go.Bar(
                    x=df_etf.index, y=df_etf['Volume'], name='Volume Traded',
                    showlegend=False), row=2, col=1)
                figOHLC_etf.update_layout(xaxis_rangeslider_visible=False)
                figOHLC_etf.update_layout(showlegend=False)
                return figOHLC_etf, df_etf
            
            
            st.subheader('D. Exchange Traded & Index Funds')
            etfselect = st.selectbox("Please select ETF here!", etflist)
            figOHLC_etf, df_etf = etf(etfselect)
            st.plotly_chart(figOHLC_etf, use_container_width=True)
            st.write("   ----   ")
            
            
            @st.cache_resource
            def data_mkt():
                df_mk = pd.DataFrame()
                df_mk['SENSEX'] = df_BSE['Close']
                df_mk['NSEI'] = df_NSEI['Close']
                df_mk['DAX'] = df_DAX['Close']
                df_mk['CAC'] = df_CAC['Close']
                df_mk['SPX'] = df_SPX['Close']
                df_mk['FTSE'] = df_FTSE['Close']
                df_mk['N225'] = df_tyo['Close']
                fig_mkt = go.Figure()
                fig_mkt.add_trace(go.Scatter(x=df_NSEI.index, y=df_NSEI['Close'],
                                             mode='lines', name='NSEI'))
                fig_mkt.add_trace(go.Scatter(x=df_BSE.index, y=df_BSE['Close'],
                                             mode='lines', name='SENSEX'))
                fig_mkt.add_trace(go.Scatter(x=df_DAX.index, y=df_DAX['Close'],
                                             mode='lines', name='DAX'))
                fig_mkt.add_trace(go.Scatter(x=df_CAC.index, y=df_CAC['Close'],
                                             mode='lines', name='CAC'))
                fig_mkt.add_trace(go.Scatter(x=df_SPX.index, y=df_SPX['Close'],
                                             mode='lines', name='SPX'))
                fig_mkt.add_trace(go.Scatter(x=df_DJIA.index, y=df_DJIA['Close'],
                                             mode='lines', name='DJIA'))
                fig_mkt.add_trace(go.Scatter(x=df_FTSE.index, y=df_FTSE['Close'],
                                             mode='lines', name='FTSE'))
                fig_mkt.add_trace(go.Scatter(x=df_tyo.index, y=df_tyo['Close'],
                                             mode='lines', name='N225'))
                fig_mkt.update_xaxes(visible=True, showticklabels=True)
                fig_mkt.update_yaxes(visible=True, showticklabels=True)
                return df_mk, fig_mkt
            
            
            df_mk, fig_mkt = data_mkt()
            
            
            curr_list = pd.read_csv(f'{direc}/pages/appdata/currency_list.csv')['Symbol']
            
            
            @st.cache_resource
            def currency(currency_selected):
                curr = yf.Ticker(currency_selected)
                currency_df = curr.history(period='2y')
            #   currency_df.to_csv(f'{direc}/pages/appdata/OHLC/{currency_selected}.csv')
                currency_df1 = currency_df.filter(['Open', 'High', 'Low', 'Close'], axis=1)
                fig_currency1 = go.Figure()
                fig_currency1.add_trace(go.Ohlc(x=currency_df1.index,
                                                open=currency_df1["Open"],
                                                high=currency_df1["High"],
                                                low=currency_df1["Low"],
                                                close=currency_df1["Close"]))
                fig_currency1.update_xaxes(visible=True, showticklabels=True)
                fig_currency1.update_yaxes(title='Exchange Ratio', visible=True,
                                           showticklabels=True)
                fig_currency1.update_layout(xaxis_rangeslider_visible=False, height=360)
                return currency_df1, fig_currency1
            
            
            st.header('E. Currencies')
            currency_selected = st.selectbox("Select Currency Pair", curr_list)
            currency_df1, fig_currency1 = currency(currency_selected)
            c11, c12 = st.columns([1, 4])
            with c11:
                with st.expander("Get the data here!"):
                    st.write(currency_df1)
                st.metric("Exchange Rate:", currency_df1['Close'].iloc[-1].round(2))
            with c12:
                st.plotly_chart(fig_currency1)
            st.write("  --------  ")
            
            st.header("F. Market Volatility Index")
            df_vix, fig_vix = vix()
            l_vix = df_vix.iloc[-1]
            st.metric("Market VIX", l_vix['Close'].round(2))
            st.plotly_chart(fig_vix, use_container_width=True)
            st.write("  --------  ")
            df_treasury, fig_treasury = treasury()
            l_ustreasury = df_treasury['Close'].iloc[-1]
            
            bn1, bn2, bn3 = st.columns([2, 1, 1])
            with bn1:
                st.header("G. Treasury Yield Rates")
                st.caption('''Estimate the real risk-free rate >>
                           Yield of the Treasury bond - Inflation Rate''')
            with bn2:
                st.metric("US Treasury", l_ustreasury.round(2))
            with bn3:
                st.write(' ')
            
            tr1, tr2 = st.tabs(["US Treasury", "Reserve Bank of India"])
            with tr1:
                st.plotly_chart(fig_treasury, use_container_width=True)
            with tr2:
                st.write(' ')

st.write("  --------  ")
column1, column2, column3, column4, column5 = st.columns([1, 1, 1, 2, 1])
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
ft1, ft2, ft3 = st.columns([1, 5, 1])
with ft1:
    st.write(" ")
with ft2:
    st.caption(""": | 2023 - 2024 | All Rights Resrved  ©  Ledgr Inc. |
               www.alphaLedgr.com | alphaLedgr Technologies Ltd. :""")
with ft3:
    st.write(" ")
