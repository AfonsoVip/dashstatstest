import streamlit as st
import base64
import pandas as pd
import numpy as np
import math as mt
import numbers
import plotly.subplots as sp
import plotly.graph_objs as go
import openpyxl
import requests

pd.options.mode.chained_assignment = None  


from io import StringIO
from PIL import Image
from io import BytesIO

logo_url = "https://github.com/AfonsoVip/dashstatstest/blob/master/logo.png?raw=true"

response = requests.get(logo_url)
logo = Image.open(BytesIO(response.content))



buffered_logo = BytesIO()
logo.save(buffered_logo, format="PNG")
logo_b64 = base64.b64encode(buffered_logo.getvalue()).decode()

st.set_page_config(layout="wide")


st.sidebar.markdown(
    f'<div class="top-right"><img src="data:image/png;base64,{logo_b64}"width="200"/><br><span style="color:white; font-size: 16px;</div>',
    unsafe_allow_html=True,
)

st.sidebar.title("Configuration")
threshold = st.sidebar.slider("Select a threshold", 0.00, 2.00, 0.00, 0.01,format='%2f')
threshold_decimal = threshold / 100

uploaded_file = st.sidebar.file_uploader("Drag and drop your file", type=['csv','xlsx','xlsm'])

submit_button = st.sidebar.button("Submit")

st.markdown(
    f'<div class="top-right" style="color:#3dfd9f">Defined Threshold: <strong style="color:#ffffff">{threshold}%</strong></div>',
    unsafe_allow_html=True,
)

def networth_evolution(df):
    fig = sp.make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Create a dictionary that maps the old strategy names to the new names
    name_mapping = {
        'NW 2STEPS LONG NO THRESHOLD LAST HOUR': 'Trading Strategy',
        'NW 2STEPS LONG WITH THRESHOLD LAST HOUR': 'Low Exposure Strategy',
        'NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR': 'High Exposure Strategy',
        'btc hold LAST HOUR': 'Hold BTC Strategy'
    }

    # Looping through each strategy and add a time series to the subplot
    strategies = list(name_mapping.keys())
    colors = ['#3dfd9f', '#66ffff', '#1d98e3', '#c00000']
    for i, strategy in enumerate(strategies):
        new_name = name_mapping[strategy]
        trace = go.Scatter(x=last_hour['StartTime'], y=last_hour[strategy], name=new_name, line=dict(color=colors[i]))
        fig.add_trace(trace)

    # Customizing the layout of the subplot
    fig.update_layout(
        width=1500,    
        height=600,
        title=dict(text='Networth Evolution (ARMS VS HOLD)', font=dict(color='#3dfd9f')),
        hovermode='x unified',
        legend=dict(x=0.7, y=0.99, bordercolor='#333333', font=dict(color='#fff')),
        font=dict(family='Calibri', size=12, color='#666'),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0)',
            linewidth=4,
            tickformat="%b %Y",
            dtick="M1",  
            tick0=df['StartTime'].min(), 
            nticks=20           
        ),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0)', linewidth=4,tickprefix='$',tickformat='.0f')
    )

    return fig


def networth_evolution_each_day(df):
    fig = sp.make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    # Create a dictionary that maps the old strategy names to the new names
    name_mapping = {
        'if started with NW 2STEPS LONG NO THRESHOLD LAST HOUR': 'Trading Strategy',
        'if started with NW 2STEPS LONG WITH THRESHOLD LAST HOUR': 'Low Exposure Strategy',
        'if started with NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR': 'High Exposure Strategy',
        'if started with btc hold LAST HOUR': 'If started with Hold BTC Strategy'
    }

    # Looping through each strategy and add a time series to the subplot
    strategies = list(name_mapping.keys())
    colors = ['#3dfd9f', '#66ffff', '#1d98e3', '#c00000']
    for i, strategy in enumerate(strategies):
        new_name = name_mapping[strategy]
        trace = go.Scatter(x=last_hour['StartTime'], y=last_hour[strategy], name=new_name, line=dict(color=colors[i]))
        fig.add_trace(trace)

    # Customizing the layout of the subplot
    fig.update_layout(
        width=1500,
        height=600,
        title=dict(text='Networth Evolution Today if started on this day (ARMS VS HOLD)', font=dict(color='#3dfd9f')),
        hovermode='x unified',
        legend=dict(x=0.6, y=0.99, bordercolor='#333333', font=dict(color='#fff')),
        font=dict(family='Calibri', size=12, color='#666'),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0)',
            linewidth=4,
            tickformat="%b %Y",
            dtick="M1",  
            tick0=df['StartTime'].min(), 
            nticks=20           
        ),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0)', linewidth=4,tickprefix='$',tickformat='.0f')
    )

    return fig

@st.cache
def threshold_summary(thresholds,df):
    results_dict = {'threshold': [], 'Low Exposure Strategy': [], 'High Exposure Strategy': []}
    for threshold in thresholds:
        result = automatizev2(df_start['StartTime'], df_start['Price Open'], df_start['Price Close'], df_start['Prediction'], threshold)
        nw_2_steps_long_with_threshold = result['NW 2STEPS LONG WITH THRESHOLD'].iloc[-1]
        nw_2_steps_long_with_threshold_and_selective_sell = result['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[-1]
        results_dict['threshold'].append(threshold)
        results_dict['Low Exposure Strategy'].append(nw_2_steps_long_with_threshold)
        results_dict['High Exposure Strategy'].append(nw_2_steps_long_with_threshold_and_selective_sell)

    results_df = pd.DataFrame(results_dict)
    results_df.set_index('threshold', inplace=True)
    results_df.index = results_df.index * 100
    return results_df

@st.cache
def automatizev2(starttime,price_open,price_close,prediction,threshold):
 

    initial_df['signal'] = np.where(initial_df['Prediction'] / initial_df['Price Close'] - 1 > 0, 1, -1)

    pct_change = initial_df['Prediction'] / initial_df['Price Close'] - 1

    initial_df['buy/hold/sell'] = np.where(pct_change > threshold, 1,
                        np.where(pct_change < -threshold, -1, 0))
    initial_df['buy/hold/sell with selective sell'] = 0  

    for index, row in initial_df.iterrows():
        if index == 1: 
            if (row['Prediction'] / row['Price Close'] - 1) > threshold:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = 1 
            else:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = 0  
        elif index > 1:  
            if (row['Prediction'] / row['Price Close'] - 1) > threshold:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = 1 
            elif (row['Prediction'] / row['Price Close'] - 1) < -threshold and (initial_df.at[index-1, 'Prediction'] / initial_df.at[index-1, 'Price Close'] - 1) < -threshold:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = -1 
            else:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = 0  
    initial_df['in/out 2 STEPS LONG NO THRESHOLD'] = initial_df['signal'].apply(lambda x: 1 if x == 1 else 0)

    initial_df['in/out 2STEPS LONG WITH THRESHOLD'] = [1] + [0] * (len(initial_df)-1)  

    for i in range(1, len(initial_df)):
        if (initial_df.at[i-1, 'in/out 2STEPS LONG WITH THRESHOLD'] + initial_df.at[i, 'buy/hold/sell']) > 0:
            initial_df.at[i, 'in/out 2STEPS LONG WITH THRESHOLD'] = 1  
        else:
            initial_df.at[i, 'in/out 2STEPS LONG WITH THRESHOLD'] = 0  
    initial_df['in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = [1] + [0] * (len(initial_df)-1) 

    for i in range(1, len(initial_df)):
        if (initial_df.at[i-1, 'in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] + initial_df.at[i, 'buy/hold/sell with selective sell']) > 0:
            initial_df.at[i, 'in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = 1  
        else:
            initial_df.at[i, 'in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = 0  


    initial_df['NW 2 STEPS LONG NO THRESHOLD'] = 100
    initial_df['NW 2 STEPS LONG NO THRESHOLD'][1] = initial_df['NW 2 STEPS LONG NO THRESHOLD'][0] * (1 + initial_df['in/out 2 STEPS LONG NO THRESHOLD'][0] * (initial_df['Price Close'][1] / initial_df['Price Close'][0] - 1))
    for i in range(2, len(initial_df)):
        initial_df['NW 2 STEPS LONG NO THRESHOLD'][i] = initial_df['NW 2 STEPS LONG NO THRESHOLD'][i-1] * (1 + initial_df['in/out 2 STEPS LONG NO THRESHOLD'][i-1] * (initial_df['Price Close'][i] / initial_df['Price Close'][i-1] - 1))

    initial_df['NW 2 STEPS LONG NO THRESHOLD'] = initial_df['NW 2 STEPS LONG NO THRESHOLD'].round(12)

    initial_df['NW 2STEPS LONG WITH THRESHOLD'] = 100
    initial_df['NW 2STEPS LONG WITH THRESHOLD'][1] = initial_df['NW 2STEPS LONG WITH THRESHOLD'][0] * (1 + initial_df['in/out 2STEPS LONG WITH THRESHOLD'][0] * (initial_df['Price Close'][1] / initial_df['Price Close'][0] - 1))
    for i in range(2, len(initial_df)):
        initial_df['NW 2STEPS LONG WITH THRESHOLD'][i] = initial_df['NW 2STEPS LONG WITH THRESHOLD'][i-1] * (1 + initial_df['in/out 2STEPS LONG WITH THRESHOLD'][i-1] * (initial_df['Price Close'][i] / initial_df['Price Close'][i-1] - 1))

    initial_df['NW 2STEPS LONG WITH THRESHOLD'] = initial_df['NW 2STEPS LONG WITH THRESHOLD'].round(12)

    initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = 100
    initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][1] = initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][0] * (1 + initial_df['in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][0] * (initial_df['Price Close'][1] / initial_df['Price Close'][0] - 1))
    for i in range(2, len(initial_df)):
        initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][i] = initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][i-1] * (1 + initial_df['in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][i-1] * (initial_df['Price Close'][i] / initial_df['Price Close'][i-1] - 1))

    initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].round(7)

    return initial_df

@st.cache
def automatize(starttime,price_open,price_close,prediction,threshold):
 

    initial_df['signal'] = np.where(initial_df['Prediction'] / initial_df['Price Close'] - 1 > 0, 1, -1)

    pct_change = initial_df['Prediction'] / initial_df['Price Close'] - 1

    initial_df['buy/hold/sell'] = np.where(pct_change > threshold, 1,
                        np.where(pct_change < -threshold, -1, 0))
    initial_df['buy/hold/sell with selective sell'] = 0  

    for index, row in initial_df.iterrows():
        if index == 1: 
            if (row['Prediction'] / row['Price Close'] - 1) > threshold:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = 1 
            else:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = 0  
        elif index > 1:  
            if (row['Prediction'] / row['Price Close'] - 1) > threshold:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = 1 
            elif (row['Prediction'] / row['Price Close'] - 1) < -threshold and (initial_df.at[index-1, 'Prediction'] / initial_df.at[index-1, 'Price Close'] - 1) < -threshold:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = -1 
            else:
                initial_df.at[index, 'buy/hold/sell with selective sell'] = 0  
    initial_df['in/out 2 STEPS LONG NO THRESHOLD'] = initial_df['signal'].apply(lambda x: 1 if x == 1 else 0)

    initial_df['in/out 2STEPS LONG WITH THRESHOLD'] = [1] + [0] * (len(initial_df)-1)  

    for i in range(1, len(initial_df)):
        if (initial_df.at[i-1, 'in/out 2STEPS LONG WITH THRESHOLD'] + initial_df.at[i, 'buy/hold/sell']) > 0:
            initial_df.at[i, 'in/out 2STEPS LONG WITH THRESHOLD'] = 1  
        else:
            initial_df.at[i, 'in/out 2STEPS LONG WITH THRESHOLD'] = 0  
    initial_df['in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = [1] + [0] * (len(initial_df)-1) 

    for i in range(1, len(initial_df)):
        if (initial_df.at[i-1, 'in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] + initial_df.at[i, 'buy/hold/sell with selective sell']) > 0:
            initial_df.at[i, 'in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = 1  
        else:
            initial_df.at[i, 'in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = 0  


    initial_df['NW 2 STEPS LONG NO THRESHOLD'] = 100
    initial_df['NW 2 STEPS LONG NO THRESHOLD'][1] = initial_df['NW 2 STEPS LONG NO THRESHOLD'][0] * (1 + initial_df['in/out 2 STEPS LONG NO THRESHOLD'][0] * (initial_df['Price Close'][1] / initial_df['Price Close'][0] - 1))
    for i in range(2, len(initial_df)):
        initial_df['NW 2 STEPS LONG NO THRESHOLD'][i] = initial_df['NW 2 STEPS LONG NO THRESHOLD'][i-1] * (1 + initial_df['in/out 2 STEPS LONG NO THRESHOLD'][i-1] * (initial_df['Price Close'][i] / initial_df['Price Close'][i-1] - 1))

    initial_df['NW 2 STEPS LONG NO THRESHOLD'] = initial_df['NW 2 STEPS LONG NO THRESHOLD'].round(12)

    initial_df['NW 2STEPS LONG WITH THRESHOLD'] = 100
    initial_df['NW 2STEPS LONG WITH THRESHOLD'][1] = initial_df['NW 2STEPS LONG WITH THRESHOLD'][0] * (1 + initial_df['in/out 2STEPS LONG WITH THRESHOLD'][0] * (initial_df['Price Close'][1] / initial_df['Price Close'][0] - 1))
    for i in range(2, len(initial_df)):
        initial_df['NW 2STEPS LONG WITH THRESHOLD'][i] = initial_df['NW 2STEPS LONG WITH THRESHOLD'][i-1] * (1 + initial_df['in/out 2STEPS LONG WITH THRESHOLD'][i-1] * (initial_df['Price Close'][i] / initial_df['Price Close'][i-1] - 1))

    initial_df['NW 2STEPS LONG WITH THRESHOLD'] = initial_df['NW 2STEPS LONG WITH THRESHOLD'].round(12)

    initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = 100
    initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][1] = initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][0] * (1 + initial_df['in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][0] * (initial_df['Price Close'][1] / initial_df['Price Close'][0] - 1))
    for i in range(2, len(initial_df)):
        initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][i] = initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][i-1] * (1 + initial_df['in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'][i-1] * (initial_df['Price Close'][i] / initial_df['Price Close'][i-1] - 1))

    initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].round(7)

    initial_df['btc hold'] = 100
    for i in range(1, len(initial_df)):
        initial_df['btc hold'][i] = initial_df['btc hold'][i-1] * initial_df['Price Close'][i] / initial_df['Price Close'][i-1]

    initial_df['btc hold'] = initial_df['btc hold'].round(12)

    last_value_nw_2_steps_long_with_no_threshold = initial_df['NW 2 STEPS LONG NO THRESHOLD'].iloc[-1]
    last_value_nw_2_steps_long_with_threshold = initial_df['NW 2STEPS LONG WITH THRESHOLD'].iloc[-1]
    last_value_nw_2_steps_long_with_threshold_and_selective_sell = initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[-1]
    last_value_btc_hold_networh_today_if_started_on_this_day = initial_df['btc hold'].iloc[-1]

    initial_df['AMRS with NO threshold networth today if started on this day'] = (last_value_nw_2_steps_long_with_no_threshold / initial_df['NW 2 STEPS LONG NO THRESHOLD']) * 100
    initial_df['AMRS with threshold networth today if started on this day'] = (last_value_nw_2_steps_long_with_threshold / initial_df['NW 2STEPS LONG WITH THRESHOLD']) * 100
    initial_df['AMRSwith threshold and ss networth today if started on this day'] = (last_value_nw_2_steps_long_with_threshold_and_selective_sell / initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'])*100
    initial_df['BTC HOLD networth today if started on this day'] = (last_value_btc_hold_networh_today_if_started_on_this_day / initial_df['btc hold']) * 100


    df21 = filter_2021_df(initial_df)
    df22 = filter_2022_df(initial_df)

    mid_value_nw_2_steps_long_with_no_threshold = df21['NW 2 STEPS LONG NO THRESHOLD'].iloc[-1]
    mid_value_nw_2_steps_long_with_threshold = df21['NW 2STEPS LONG WITH THRESHOLD'].iloc[-1]
    mid_value_nw_2_steps_long_with_threshold_and_selective_sell = df21['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[-1]
    mid_value_btc_hold_networh_today_if_started_on_this_day = df21['btc hold'].iloc[-1]


    first_half = (mid_value_nw_2_steps_long_with_no_threshold/ df21['NW 2 STEPS LONG NO THRESHOLD']) * 100
    input_value = (mid_value_nw_2_steps_long_with_no_threshold/mid_value_nw_2_steps_long_with_no_threshold) * 100
    second_half = (last_value_nw_2_steps_long_with_no_threshold/df22['NW 2 STEPS LONG NO THRESHOLD']) * 100

    first_half = pd.Series(first_half)
    input_value = pd.Series(input_value)
    second_half = pd.Series(second_half)
    concated_series_1 = pd.concat([first_half,input_value,second_half]).reset_index(drop=True)


    initial_df['AMRS with NO threshold networth today if started on the first day of each year'] = concated_series_1

    first_half_2 = (mid_value_nw_2_steps_long_with_threshold/ df21['NW 2STEPS LONG WITH THRESHOLD']) * 100
    input_value_2 = (mid_value_nw_2_steps_long_with_threshold/mid_value_nw_2_steps_long_with_threshold) * 100
    second_half_2 = (last_value_nw_2_steps_long_with_threshold/df22['NW 2STEPS LONG WITH THRESHOLD']) * 100

    first_half_2 = pd.Series(first_half_2)
    input_value_2= pd.Series(input_value_2)
    second_half_2 = pd.Series(second_half_2)
    concated_series_2 = pd.concat([first_half_2,input_value_2,second_half_2]).reset_index(drop=True)


    initial_df['AMRS with threshold networth today if started on the first day of each year'] = concated_series_2

    first_half_3 = (mid_value_nw_2_steps_long_with_threshold_and_selective_sell/ df21['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL']) * 100
    input_value_3 = (mid_value_nw_2_steps_long_with_threshold_and_selective_sell/mid_value_nw_2_steps_long_with_threshold_and_selective_sell) * 100
    second_half_3 = (last_value_nw_2_steps_long_with_threshold_and_selective_sell/df22['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL']) * 100

    first_half_3 = pd.Series(first_half_3)
    input_value_3 = pd.Series(input_value_3)
    second_half_3 = pd.Series(second_half_3)
    concated_series_3 = pd.concat([first_half_3,input_value_3,second_half_3]).reset_index(drop=True)


    initial_df['AMRSwith threshold and ss networth today if started on the first day of each year'] = concated_series_3

    first_half_4 = (mid_value_btc_hold_networh_today_if_started_on_this_day/ df21['btc hold']) * 100
    input_value_4 = (mid_value_btc_hold_networh_today_if_started_on_this_day/mid_value_btc_hold_networh_today_if_started_on_this_day) * 100
    second_half_4 = (last_value_btc_hold_networh_today_if_started_on_this_day/df22['btc hold']) * 100

    first_half_4 = pd.Series(first_half_4)
    input_value_4 = pd.Series(input_value_4)
    second_half_4 = pd.Series(second_half_4)
    concated_series_4 = pd.concat([first_half_4,input_value_4,second_half_4]).reset_index(drop=True)


    initial_df['BTC HOLD networth today if started on the first day of each year']= concated_series_4

    initial_df['AMRS with NO threshold networth today if started on this day'].round(10)
    initial_df['AMRS with threshold networth today if started on this day'].round(10)
    initial_df['AMRSwith threshold and ss networth today if started on this day'].round(11)  
    initial_df['BTC HOLD networth today if started on this day'].round(12)

    initial_df['return of portfolio 2STEPS LONG WITH NO THRESHOL AND SELECTIVE SELL'] = initial_df['NW 2 STEPS LONG NO THRESHOLD'].pct_change()
    initial_df['return of portfolio 2STEPS LONG WITH NO THRESHOL AND SELECTIVE SELL'].iloc[0] = 0

    initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD'] = initial_df['NW 2STEPS LONG WITH THRESHOLD'].pct_change()
    initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD'].iloc[0] = 0
    initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD'].round(16)

    initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = initial_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].pct_change()
    initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[0] = 0
    initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] .round(16)

    initial_df['return of btc'] = initial_df['btc hold'].pct_change()
    initial_df['return of btc'].iloc[0] = 0
    initial_df['return of btc'] = initial_df['return of btc'] .round(16)



    actual_signal = initial_df['return of btc'].apply(lambda x: 0 if x == 0 else (1 if x > 0 else -1))
    accuracy_signals = [0] + (actual_signal.iloc[1:].reset_index(drop=True) == initial_df['signal'].iloc[:-1].reset_index(drop=True)).astype(int).tolist()
    initial_df['accuracy signals'] = accuracy_signals
    initial_df['accuracy strategy with threshold'] = ((initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD'] > 0) & (initial_df['return of btc'] > 0) | (initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD'] == 0) & (initial_df['return of btc'] < 0)).astype(int)
    initial_df['accuracy strategy with threshold and selective sell'] = ((initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] > 0) & (initial_df['return of btc'] > 0) | (initial_df['return of portfolio 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] == 0) & (initial_df['return of btc'] < 0)).astype(int)

    initial_df['BETTER ARMS with NO threshold'] = (initial_df['AMRS with NO threshold networth today if started on this day'] > initial_df['BTC HOLD networth today if started on this day']).astype(int)
    initial_df['BETTER ARMS with threshold'] = (initial_df['AMRS with threshold networth today if started on this day'] > initial_df['BTC HOLD networth today if started on this day']).astype(int)
    initial_df['BETTER ARMS  with threshold and ss'] = (initial_df['AMRSwith threshold and ss networth today if started on this day'] > initial_df['BTC HOLD networth today if started on this day']).astype(int)

    return initial_df

def last_hour_df(df):
    last_hour_df = main_df[main_df['StartTime'].apply(lambda x: x.hour) == 23]
    last_hour_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR'] = last_hour_df['StartTime'].apply(lambda x: main_df.loc[main_df['StartTime'] == x, 'NW 2 STEPS LONG NO THRESHOLD'].iloc[0])
    last_hour_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR'] = last_hour_df['StartTime'].apply(lambda x: main_df.loc[main_df['StartTime'] == x, 'NW 2STEPS LONG WITH THRESHOLD'].iloc[0])
    last_hour_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR'] = last_hour_df['StartTime'].apply(lambda x: main_df.loc[main_df['StartTime'] == x, 'NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[0])
    last_hour_df['btc hold LAST HOUR'] = last_hour_df['StartTime'].apply(lambda x: main_df.loc[main_df['StartTime'] == x, 'btc hold'].iloc[0])
    last_hour_df['NW 2STEPS LONG NO THRESHOLD RATIO'] = (last_hour_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR'] / last_hour_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR'].shift(1)) - 1
    last_hour_df['NW 2STEPS LONG NO THRESHOLD RATIO'].iloc[0] = last_hour_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR'].iloc[0] / 100 - 1
    last_hour_df['NW 2STEPS LONG WITH THRESHOLD RATIO'] = (last_hour_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR'] / last_hour_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR'].shift(1)) - 1
    last_hour_df['NW 2STEPS LONG WITH THRESHOLD RATIO'].iloc[0] = last_hour_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR'].iloc[0] / 100 - 1
    last_hour_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'] = (last_hour_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR'] / last_hour_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR'].shift(1)) - 1
    last_hour_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].iloc[0] = last_hour_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR'].iloc[0] / 100 - 1
    last_hour_df['btc hold RATIO'] = (last_hour_df['btc hold LAST HOUR'] / last_hour_df['btc hold LAST HOUR'].shift(1)) - 1
    last_hour_df['btc hold RATIO'].iloc[0] = last_hour_df['btc hold LAST HOUR'].iloc[0] / 100 - 1
    last_hour_df = last_hour_df[['StartTime','NW 2STEPS LONG NO THRESHOLD LAST HOUR','NW 2STEPS LONG WITH THRESHOLD LAST HOUR','NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR','btc hold LAST HOUR','NW 2STEPS LONG NO THRESHOLD RATIO','NW 2STEPS LONG WITH THRESHOLD RATIO','NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO','btc hold RATIO']]

    last_value_nw_2_steps_long_with_no_threshold = last_hour_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR'].iloc[-1]
    last_value_nw_2_steps_long_with_threshold = last_hour_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR'].iloc[-1]
    last_value_nw_2_steps_long_with_threshold_and_selective_sell = last_hour_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR'].iloc[-1]
    last_value_btc_hold_networh_today_if_started_on_this_day = last_hour_df['btc hold LAST HOUR'].iloc[-1]

    last_hour_df['if started with NW 2STEPS LONG NO THRESHOLD LAST HOUR'] = last_value_nw_2_steps_long_with_no_threshold/ last_hour_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR'] * 100
    last_hour_df['if started with NW 2STEPS LONG WITH THRESHOLD LAST HOUR'] = last_value_nw_2_steps_long_with_threshold/ last_hour_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR'] * 100
    last_hour_df['if started with NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR'] = last_value_nw_2_steps_long_with_threshold_and_selective_sell/ last_hour_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR'] * 100
    last_hour_df['if started with btc hold LAST HOUR'] = last_value_btc_hold_networh_today_if_started_on_this_day/ last_hour_df['btc hold LAST HOUR'] * 100

    last_hour_df['END OF MONTH'] = last_hour_df['StartTime'].apply(lambda x: x.replace(day=1, hour=23, minute=0, second=0, microsecond=0) + pd.offsets.MonthEnd(1))
    

    return last_hour_df


def last_hour_and_day_df(df):
    last_days = pd.date_range(start=main_df['StartTime'].min().floor('d'), end=main_df['StartTime'].max().ceil('d'), freq='M').values
    last_days = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in last_days]
    last_hour_and_day_df = main_df[(main_df['StartTime'].dt.strftime('%Y-%m-%d').isin(last_days)) & (main_df['StartTime'].dt.hour == 23)]
    last_hour_and_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY'] = last_hour_and_day_df['StartTime'].apply(lambda x: main_df.loc[main_df['StartTime'] == x, 'NW 2 STEPS LONG NO THRESHOLD'].iloc[0])
    last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY'] = last_hour_and_day_df['StartTime'].apply(lambda x: main_df.loc[main_df['StartTime'] == x, 'NW 2STEPS LONG WITH THRESHOLD'].iloc[0])
    last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY'] = last_hour_and_day_df['StartTime'].apply(lambda x: main_df.loc[main_df['StartTime'] == x, 'NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[0])
    last_hour_and_day_df['btc hold LAST HOUR AND DAY'] = last_hour_and_day_df['StartTime'].apply(lambda x: df.loc[df['StartTime'] == x, 'btc hold'].iloc[0])
    last_hour_and_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'] = (last_hour_and_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY'] / last_hour_and_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY'].shift(1)) - 1
    last_hour_and_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].iloc[0] = last_hour_and_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY'].iloc[0] / 100 - 1
    last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'] = (last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY'] / last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY'].shift(1)) - 1
    last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].iloc[0] = last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY'].iloc[0] / 100 - 1
    last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'] = (last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY'] / last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY'].shift(1)) - 1
    last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].iloc[0] = last_hour_and_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY'].iloc[0] / 100 - 1
    last_hour_and_day_df['btc hold LAST HOUR AND DAY RATIO'] = (last_hour_and_day_df['btc hold LAST HOUR AND DAY'] / last_hour_and_day_df['btc hold LAST HOUR AND DAY'].shift(1)) - 1
    last_hour_and_day_df['btc hold LAST HOUR AND DAY RATIO'].iloc[0] = last_hour_and_day_df['btc hold LAST HOUR AND DAY'].iloc[0] / 100 - 1


    last_hour_and_day_df['volatility NW 2STEPS LONG WITH NO THRESHOLD'] = last_hour_and_day_df.apply(lambda x: np.std(last_hour.loc[last_hour['END OF MONTH'] == x['StartTime'], 'NW 2STEPS LONG NO THRESHOLD RATIO'] ), axis=1) * 100
    last_hour_and_day_df['volatility NW 2STEPS LONG WITH THRESHOLD'] = last_hour_and_day_df.apply(lambda x: np.std(last_hour.loc[last_hour['END OF MONTH'] == x['StartTime'], 'NW 2STEPS LONG WITH THRESHOLD RATIO'] ), axis=1) * 100
    last_hour_and_day_df['volatility NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = last_hour_and_day_df.apply(lambda x: np.std(last_hour.loc[last_hour['END OF MONTH'] == x['StartTime'], 'NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'] ), axis=1) * 100
    last_hour_and_day_df['volatility btc hold'] = last_hour_and_day_df.apply(lambda x: np.std(last_hour.loc[last_hour['END OF MONTH'] == x['StartTime'], 'btc hold RATIO'] ), axis=1) * 100
    last_hour_and_day_df = last_hour_and_day_df[['StartTime','NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY','NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY','NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY','btc hold LAST HOUR AND DAY','NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO','NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO','NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO','btc hold LAST HOUR AND DAY RATIO',
                                                 'volatility NW 2STEPS LONG WITH NO THRESHOLD','volatility NW 2STEPS LONG WITH THRESHOLD','volatility NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL','volatility btc hold']]
    return last_hour_and_day_df


def last_day_of_the_year_last_hour(df):
    last_day_filter = (df['StartTime'].dt.month == 12) & (df['StartTime'].dt.day == 31) & (df['StartTime'].dt.hour == 23)
    last_day_df = df[last_day_filter]
    last_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR'] = last_day_df['StartTime'].apply(lambda x: df.loc[df['StartTime'] == x, 'NW 2 STEPS LONG NO THRESHOLD'].iloc[0])
    last_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR'] = last_day_df['StartTime'].apply(lambda x: df.loc[df['StartTime'] == x, 'NW 2STEPS LONG WITH THRESHOLD'].iloc[0])
    last_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR'] = last_day_df['StartTime'].apply(lambda x: df.loc[df['StartTime'] == x, 'NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[0])
    last_day_df['btc hold'] = last_day_df['StartTime'].apply(lambda x: df.loc[df['StartTime'] == x, 'btc hold'].iloc[0])

    last_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'] = (last_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR'] / last_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR'].shift(1)) - 1  
    last_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[0] = last_day_df['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR'].iloc[0] / 100 - 1

    last_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'] = (last_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR'] / last_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR'].shift(1)) - 1
    last_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[0] = last_day_df['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR'].iloc[0] / 100 - 1

    last_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR RATIO'] = (last_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR'] / last_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR'].shift(1)) - 1
    last_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[0] = last_day_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR'].iloc[0] / 100 - 1
    
    last_day_df['btc hold RATIO'] = (last_day_df['btc hold'] / last_day_df['btc hold'].shift(1)) - 1
    last_day_df['btc hold RATIO'].iloc[0] = last_day_df['btc hold'].iloc[0] / 100 - 1
    last_day_df = last_day_df[['StartTime','NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR','NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR','NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR','btc hold','NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO','NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO','NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR RATIO','btc hold RATIO']]
    return last_day_df


def format_percentage1(col):
    return col.apply(lambda x: safe_round_and_format(x))

def safe_round_and_format(x):
    try:
        return str(round(x)) + '%'
    except (TypeError, ValueError):
        return ''

def return_volatility(df):

    returns = pd.DataFrame({
        'StartTime': last_hour_day['StartTime'].to_list(), 
        '2STEPS LONG WITH NO THRESHOLD': format_percentage1(pd.Series((last_hour_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO']*100).to_list())),
        '2STEPS LONG WITH THRESHOLD': format_percentage1(pd.Series((last_hour_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO']*100).to_list())),
        '2STEPS LONG WITH THRESHOLD and ss': format_percentage1(pd.Series((last_hour_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO']*100).to_list())),
        'btc hold': format_percentage1(pd.Series((last_hour_day['btc hold LAST HOUR AND DAY RATIO']*100).to_list()))
    })

    volatilities = pd.DataFrame({
        'StartTime': last_hour_day['StartTime'].to_list(), 
        '2STEPS LONG WITH NO THRESHOLD': format_percentage1(pd.Series(last_hour_day['volatility NW 2STEPS LONG WITH NO THRESHOLD'].to_list())),
        '2STEPS LONG WITH THRESHOLD': format_percentage1(pd.Series(last_hour_day['volatility NW 2STEPS LONG WITH THRESHOLD'].to_list())),
        '2STEPS LONG WITH THRESHOLD and ss': format_percentage1(pd.Series(last_hour_day['volatility NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].to_list())),
        'btc hold': format_percentage1(pd.Series(last_hour_day['volatility btc hold'].to_list()))
    })

    return_vol = pd.merge(returns, volatilities, on="StartTime", suffixes=["_return", "_volatility"])

    return_vol = return_vol[['StartTime','2STEPS LONG WITH NO THRESHOLD_return','2STEPS LONG WITH NO THRESHOLD_volatility','2STEPS LONG WITH THRESHOLD_return','2STEPS LONG WITH THRESHOLD_volatility',
                            '2STEPS LONG WITH THRESHOLD and ss_return','2STEPS LONG WITH THRESHOLD and ss_volatility','btc hold_return','btc hold_volatility']]

    columns = pd.MultiIndex.from_tuples([('StartTime', ''),('Trading Strategy', 'return'), ('', 'volatility'),
                                        ('Low Exposure Strategy', 'return'), ('', 'volatility'),
                                        ('High Exposure Strategy', 'return'), ('', 'volatility'),
                                        ('Hold BTC Strategy', 'return'), ('', 'volatility')])

    return_vol.columns = columns
    return_vol = return_vol.reset_index(drop = True)
    return_vol.index = return_vol.index + 1
    return_vol = return_vol.transpose()


    return return_vol

def format_percentage(col):
    return col.apply(lambda x: safe_round_and_format(x))


def important_scores(df,last_hour,last_day_of_the_year):

    # NW 2 STEPS LONG NO THRESHOLD

    return_s_first = ((last_day_of_the_year['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR'].iloc[-1] / 100) - 1) * 100
    volatility_first = last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].std() * 100
    annualized_sharpe_ratio_first = ((last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].mean() - 0.02/365) / (last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].std()) * mt.sqrt(365)) 
    accuracy_strategy_first = df['accuracy signals'].mean() * 100
    accuracy_signals_first = df['accuracy signals'].mean() * 100 
    better_arms_first = df['BETTER ARMS with NO threshold'].mean() * 100 
    difference_from_ATH_first = (df['NW 2 STEPS LONG NO THRESHOLD'].iloc[-1] / df['NW 2 STEPS LONG NO THRESHOLD'].max() - 1) * 100
    maximum_daily_gain_first = last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].max() * 100
    maximum_daily_loss_first = last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].min() * 100
    mean_monthly_return_first = last_hour_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_first = last_hour_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_first = last_hour_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].min() * 100
    return_2021_first = last_day_of_the_year['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[0] * 100 
    return_2022_first = last_day_of_the_year['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[-1] * 100
    percentagetime_in_first = df['in/out 2 STEPS LONG NO THRESHOLD'].mean() * 100
    percentagetime_buy_first = df['signal'].value_counts(normalize=True)[1] * 100
    percentagetime_hold_first = df['signal'].value_counts(normalize=True, dropna=False).get("missing_key", 0) * 100
    percentagetime_sell_first = df['signal'].value_counts(normalize=True)[-1] * 100
   

    # NW 2STEPS LONG WITH THRESHOLD 
    return_s_second = ((last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR'].iloc[-1] / 100) - 1) * 100
    volatility_second = last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].std() * 100
    annualized_sharpe_ratio_second = ((last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].mean() - 0.02/365) / (last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].std()) * mt.sqrt(365)) 
    accuracy_strategy_second = df['accuracy strategy with threshold'].mean() * 100
    accuracy_signals_second = df['accuracy signals'].mean() * 100 
    better_arms_second = df['BETTER ARMS with threshold'].mean() * 100 
    difference_from_ATH_second = (df['NW 2STEPS LONG WITH THRESHOLD'].iloc[-1] / df['NW 2STEPS LONG WITH THRESHOLD'].max() - 1) * 100
    maximum_daily_gain_second = last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].max() * 100
    maximum_daily_loss_second = last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].min() * 100
    mean_monthly_return_second = last_hour_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_second = last_hour_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_second = last_hour_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].min() * 100
    return_2021_second = last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[0] * 100 
    return_2022_second = last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[-1] * 100
    percentagetime_in_second = df['in/out 2STEPS LONG WITH THRESHOLD'].mean() * 100
    percentagetime_buy_second = df['buy/hold/sell'].value_counts(normalize=True)[1] * 100
    percentagetime_hold_second = df['buy/hold/sell'].value_counts(normalize=True)[0] * 100
    percentagetime_sell_second = df['buy/hold/sell'].value_counts(normalize=True)[-1] * 100
 

    # NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL
    return_s_third = ((last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR'].iloc[-1] / 100) - 1) * 100
    volatility_third = last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].std() * 100
    annualized_sharpe_ratio_third = ((last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].mean() - 0.02/365) / (last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].std()) * mt.sqrt(365)) 
    accuracy_strategy_third = df['accuracy strategy with threshold and selective sell'].mean() * 100
    accuracy_signals_third = df['accuracy signals'].mean() * 100
    better_arms_third = df['BETTER ARMS  with threshold and ss'].mean() * 100
    difference_from_ATH_third = (df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[-1] / df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].max() - 1) * 100
    maximum_daily_gain_third = last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].max() * 100
    maximum_daily_loss_third = last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].min() * 100
    mean_monthly_return_third = last_hour_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_third = last_hour_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_third = last_hour_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].min() * 100
    return_2021_third = last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[0] * 100 
    return_2022_third = last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[-1] * 100
    percentagetime_in_third = df['in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].mean() * 100
    percentagetime_buy_third = df['buy/hold/sell with selective sell'].value_counts(normalize=True)[1] * 100
    percentagetime_hold_third = df['buy/hold/sell with selective sell'].value_counts(normalize=True)[0] * 100
    percentagetime_sell_third = df['buy/hold/sell with selective sell'].value_counts(normalize=True)[-1] * 100
   

    #BTC HOLD
    return_s_forth = ((last_day_of_the_year['btc hold'].iloc[-1] / 100) - 1) * 100
    volatility_forth = last_hour['btc hold RATIO'].std() * 100
    annualized_sharpe_ratio_forth = ((last_hour['btc hold RATIO'].mean() - 0.02/365) / (last_hour['btc hold RATIO'].std()) * mt.sqrt(365)) * 100
    accuracy_strategy_forth = np.nan
    accuracy_signals_forth = np.nan
    better_arms_forth = np.nan
    difference_from_ATH_forth = (df['btc hold'].iloc[-1] / df['btc hold'].max() - 1) * 100
    maximum_daily_gain_forth = last_hour['btc hold RATIO'].max() * 100
    maximum_daily_loss_forth = last_hour['btc hold RATIO'].min() * 100
    mean_monthly_return_forth = last_hour_day['btc hold LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_forth = last_hour_day['btc hold LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_forth = last_hour_day['btc hold LAST HOUR AND DAY RATIO'].min() * 100
    return_2021_forth = last_day_of_the_year['btc hold RATIO'].iloc[0] * 100 
    return_2022_forth= last_day_of_the_year['btc hold RATIO'].iloc[-1] * 100
    percentagetime_in_forth = np.nan
    percentagetime_buy_forth = np.nan
    percentagetime_hold_forth = np.nan
    percentagetime_sell_forth = np.nan
    

    index_column = ['return','volatility', 'annualized sharpe ratio', 'ACCURACY strategy', 'accuracy signals', 'BETTER ARMS',
                'difference from ATH', 'maximum daily gain', 'maximum daily loss', 'mean monthly return',
                'best monthly return', 'worst month return', 'return 2021', 'return 2022', '%time in', '% buy',
                '% hold', '% sell']

    first_data = [return_s_first, volatility_first, annualized_sharpe_ratio_first, accuracy_strategy_first, accuracy_signals_first, better_arms_first, difference_from_ATH_first, maximum_daily_gain_first, maximum_daily_loss_first, mean_monthly_return_first, best_monthly_return_first, worst_monthly_return_first, return_2021_first, return_2022_first, percentagetime_in_first, percentagetime_buy_first, percentagetime_hold_first, percentagetime_sell_first]
 
    second_data = [return_s_second,volatility_second,annualized_sharpe_ratio_second,accuracy_strategy_second,accuracy_signals_second,better_arms_second,difference_from_ATH_second,maximum_daily_gain_second,
    maximum_daily_loss_second,mean_monthly_return_second,best_monthly_return_second,worst_monthly_return_second,return_2021_second,return_2022_second,percentagetime_in_second,percentagetime_buy_second,percentagetime_hold_second,
    percentagetime_sell_second]

    third_data = [return_s_third,volatility_third,annualized_sharpe_ratio_third,accuracy_strategy_third,accuracy_signals_third,better_arms_third,difference_from_ATH_third,maximum_daily_gain_third,
    maximum_daily_loss_third,mean_monthly_return_third,best_monthly_return_third,worst_monthly_return_third,return_2021_third,return_2022_third,percentagetime_in_third,percentagetime_buy_third,
    percentagetime_hold_third,percentagetime_sell_third]

    forth_data = [return_s_forth,volatility_forth,annualized_sharpe_ratio_forth,accuracy_strategy_forth,accuracy_signals_forth,better_arms_forth,difference_from_ATH_forth,maximum_daily_gain_forth,maximum_daily_loss_forth,
    mean_monthly_return_forth,best_monthly_return_forth,worst_monthly_return_forth,return_2021_forth,return_2022_forth,percentagetime_in_forth,percentagetime_buy_forth,percentagetime_hold_forth,
    percentagetime_sell_forth]

    important_scores = pd.DataFrame(columns=['NW 2STEPS LONG NO THRESHOLD'], index=index_column)
    important_scores['NW 2STEPS LONG NO THRESHOLD'] = first_data
    important_scores['NW 2STEPS LONG WITH THRESHOLD'] = second_data
    important_scores['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = third_data
    important_scores['BTC HOLD'] = forth_data

    return important_scores

def filter_2021_df(df):
    mask = (df['StartTime'] >= '2021-01-01 00:00:00') & (df['StartTime'] <= '2021-12-31 23:00:00')
    df_filter_21 = df.loc[mask]
    return df_filter_21

def filter_2022_df(df):
    mask = (df['StartTime'] >= '2022-01-01 00:00:00') & (df['StartTime'] <= '2022-12-31 23:00:00')
    df_filter_22 = df.loc[mask]
    return df_filter_22


def filter_2021_2022_df(df):
    mask = (df['StartTime'] >= '2021-01-01 00:00:00') & (df['StartTime'] <= '202-12-31 23:00:00')
    df_filter_21 = df.loc[mask]
    return df_filter_21


def filter_2021_last_hour_df(df):
    mask = (last_hour['StartTime'].dt.year == 2021) & (last_hour['StartTime'].dt.time == pd.to_datetime('23:00:00').time())
    df_filter_21_last_hour = last_hour.loc[mask]
    return df_filter_21_last_hour

def filter_2021_last_hour_last_day(df):
    mask = (last_hour_day['StartTime'].dt.year == 2021) & (last_hour_day['StartTime'].dt.time == pd.to_datetime('23:00:00').time())
    df_filter_22_last_hour = last_hour_day.loc[mask]
    return df_filter_22_last_hour

def filter_2022_last_hour_df(df):
    mask = (last_hour['StartTime'].dt.year == 2022) & (last_hour['StartTime'].dt.time == pd.to_datetime('23:00:00').time())
    df_filter_22_last_hour = last_hour.loc[mask]
    return df_filter_22_last_hour

def filter_2022_last_hour_last_day(df):
    mask = (last_hour_day['StartTime'].dt.year == 2022) & (last_hour_day['StartTime'].dt.time == pd.to_datetime('23:00:00').time())
    df_filter_22_last_hour = last_hour_day.loc[mask]
    return df_filter_22_last_hour

def important_scores_21(df,last_hour,last_day_of_the_year):

    # NW 2 STEPS LONG NO THRESHOLD
    return_2021_first = last_day_of_the_year['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[0] * 100
    volatility_first = df_21_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].std() * 100
    annualized_sharpe_ratio_first = ((df_21_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].mean() - 0.02/365) / (df_21_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].std()) * mt.sqrt(365)) 
    accuracy_strategy_first = df_21['accuracy strategy with threshold'].mean() * 100
    accuracy_signals_first = df_21['accuracy signals'].mean() * 100 
    better_arms_first = df_21['BETTER ARMS with NO threshold'].mean() * 100 
    difference_from_ATH_first = (df_21['NW 2 STEPS LONG NO THRESHOLD'].iloc[-1] / df_21['NW 2 STEPS LONG NO THRESHOLD'].max() - 1) * 100
    maximum_daily_gain_first = df_21_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].max() * 100
    maximum_daily_loss_first = df_21_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].min() * 100
    mean_monthly_return_first = df_21_last_hour_last_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_first = df_21_last_hour_last_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_first = df_21_last_hour_last_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].min() * 100
    percentagetime_in_first = df_21['in/out 2 STEPS LONG NO THRESHOLD'].mean() * 100
    percentagetime_buy_first = df_21['signal'].value_counts(normalize=True)[1] * 100
    percentagetime_hold_first = df_21['signal'].value_counts(normalize=True, dropna=False).get("missing_key", 0) * 100
    percentagetime_sell_first = df_21['signal'].value_counts(normalize=True)[-1] * 100


     # NW 2STEPS LONG WITH THRESHOLD 
    return_2021_second = last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[0] * 100 
    volatility_second = df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].std() * 100
    annualized_sharpe_ratio_second = ((df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].mean() - 0.02/365) / (df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].std()) * mt.sqrt(365)) 
    accuracy_strategy_second = df_21['accuracy strategy with threshold'].mean() * 100
    accuracy_signals_second = df_21['accuracy signals'].mean() * 100 
    better_arms_second = df_21['BETTER ARMS with threshold'].mean() * 100 
    difference_from_ATH_second = (df_21['NW 2STEPS LONG WITH THRESHOLD'].iloc[-1] / df_21['NW 2STEPS LONG WITH THRESHOLD'].max() - 1) * 100
    maximum_daily_gain_second = df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].max() * 100
    maximum_daily_loss_second = df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].min() * 100
    mean_monthly_return_second = df_21_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_second = df_21_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_second = df_21_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].min() * 100
    percentagetime_in_second = df_21['in/out 2STEPS LONG WITH THRESHOLD'].mean() * 100
    percentagetime_buy_second = df_21['buy/hold/sell'].value_counts(normalize=True)[1] * 100
    percentagetime_hold_second = df_21['buy/hold/sell'].value_counts(normalize=True)[0] * 100
    percentagetime_sell_second = df_21['buy/hold/sell'].value_counts(normalize=True)[-1] * 100

    # NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL
    return_2021_third = last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[0] * 100
    volatility_third = df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].std() * 100
    annualized_sharpe_ratio_third = ((df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].mean() - 0.02/365) / (df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].std()) * mt.sqrt(365)) 
    accuracy_strategy_third = df_21['accuracy strategy with threshold and selective sell'].mean() * 100
    accuracy_signals_third = df_21['accuracy signals'].mean() * 100
    better_arms_third = df_21['BETTER ARMS  with threshold and ss'].mean() * 100
    difference_from_ATH_third = (df_21['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[-1] / df_21['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].max() - 1) * 100
    maximum_daily_gain_third = df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].max() * 100
    maximum_daily_loss_third = df_21_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].min() * 100
    mean_monthly_return_third = df_21_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_third = df_21_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_third = df_21_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].min() * 100
    percentagetime_in_third = df_21['in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].mean() * 100
    percentagetime_buy_third = df_21['buy/hold/sell with selective sell'].value_counts(normalize=True)[1] * 100
    percentagetime_hold_third = df_21['buy/hold/sell with selective sell'].value_counts(normalize=True)[0] * 100
    percentagetime_sell_third = df_21['buy/hold/sell with selective sell'].value_counts(normalize=True)[-1] * 100

    #BTC HOLD
    return_2021_forth = last_day_of_the_year['btc hold RATIO'].iloc[0] * 100
    volatility_forth = df_21_last_hour['btc hold RATIO'].std() * 100
    annualized_sharpe_ratio_forth = ((df_21_last_hour['btc hold RATIO'].mean() - 0.02/365) / (df_21_last_hour['btc hold RATIO'].std()) * mt.sqrt(365)) * 100
    accuracy_strategy_forth = np.nan
    accuracy_signals_forth = np.nan
    better_arms_forth = np.nan
    difference_from_ATH_forth = (df_21['btc hold'].iloc[-1] / df_21['btc hold'].max() - 1) * 100
    maximum_daily_gain_forth = df_21_last_hour['btc hold RATIO'].max() * 100
    maximum_daily_loss_forth = df_21_last_hour['btc hold RATIO'].min() * 100
    mean_monthly_return_forth = df_21_last_hour_last_day['btc hold LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_forth = df_21_last_hour_last_day['btc hold LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_forth = df_21_last_hour_last_day['btc hold LAST HOUR AND DAY RATIO'].min() * 100
    percentagetime_in_forth = np.nan
    percentagetime_buy_forth = np.nan
    percentagetime_hold_forth = np.nan
    percentagetime_sell_forth = np.nan

    index_column = ['return','volatility', 'annualized sharpe ratio', 'ACCURACY strategy', 'accuracy signals', 'BETTER ARMS',
                'difference from ATH', 'maximum daily gain', 'maximum daily loss', 'mean monthly return',
                'best monthly return', 'worst month return', '%time in', '% buy',
                '% hold', '% sell']

    first_data = [return_2021_first, volatility_first, annualized_sharpe_ratio_first, accuracy_strategy_first, accuracy_signals_first, better_arms_first, difference_from_ATH_first, maximum_daily_gain_first, 
    maximum_daily_loss_first, mean_monthly_return_first, best_monthly_return_first, worst_monthly_return_first, percentagetime_in_first, percentagetime_buy_first, percentagetime_hold_first, percentagetime_sell_first]
 
    second_data = [return_2021_second,volatility_second,annualized_sharpe_ratio_second,accuracy_strategy_second,accuracy_signals_second,better_arms_second,difference_from_ATH_second,maximum_daily_gain_second,
    maximum_daily_loss_second,mean_monthly_return_second,best_monthly_return_second,worst_monthly_return_second,percentagetime_in_second,percentagetime_buy_second,percentagetime_hold_second,
    percentagetime_sell_second]

    third_data = [return_2021_third,volatility_third,annualized_sharpe_ratio_third,accuracy_strategy_third,accuracy_signals_third,better_arms_third,difference_from_ATH_third,maximum_daily_gain_third,
    maximum_daily_loss_third,mean_monthly_return_third,best_monthly_return_third,worst_monthly_return_third,percentagetime_in_third,percentagetime_buy_third,
    percentagetime_hold_third,percentagetime_sell_third]

    forth_data = [return_2021_forth,volatility_forth,annualized_sharpe_ratio_forth,accuracy_strategy_forth,accuracy_signals_forth,better_arms_forth,difference_from_ATH_forth,maximum_daily_gain_forth,maximum_daily_loss_forth,
    mean_monthly_return_forth,best_monthly_return_forth,worst_monthly_return_forth,percentagetime_in_forth,percentagetime_buy_forth,percentagetime_hold_forth,
    percentagetime_sell_forth]

    important_scores = pd.DataFrame(columns=['NW 2STEPS LONG NO THRESHOLD'], index=index_column)
    important_scores['NW 2STEPS LONG NO THRESHOLD'] = first_data
    important_scores['NW 2STEPS LONG WITH THRESHOLD'] = second_data
    important_scores['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = third_data
    important_scores['BTC HOLD'] = forth_data

    return important_scores


def important_scores_22(df,last_hour,last_day_of_the_year):

    # NW 2 STEPS LONG NO THRESHOLD
    return_2022_first = last_day_of_the_year['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[-1] * 100
    volatility_first = df_22_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].std() * 100
    annualized_sharpe_ratio_first = ((df_22_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].mean() - 0.02/365) / (df_22_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].std()) * mt.sqrt(365)) 
    accuracy_strategy_first = df_22['accuracy signals'].mean() * 100
    accuracy_signals_first = df_22['accuracy signals'].mean() * 100 
    better_arms_first = df_22['BETTER ARMS with NO threshold'].mean() * 100 
    difference_from_ATH_first = (df_22['NW 2 STEPS LONG NO THRESHOLD'].iloc[-1] / df_22['NW 2 STEPS LONG NO THRESHOLD'].max() - 1) * 100
    maximum_daily_gain_first = df_22_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].max() * 100
    maximum_daily_loss_first = df_22_last_hour['NW 2STEPS LONG NO THRESHOLD RATIO'].min() * 100
    mean_monthly_return_first = df_22_last_hour_last_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_first = df_22_last_hour_last_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_first = df_22_last_hour_last_day['NW 2STEPS LONG NO THRESHOLD LAST HOUR AND DAY RATIO'].min() * 100
    percentagetime_in_first = df_22['in/out 2 STEPS LONG NO THRESHOLD'].mean() * 100
    percentagetime_buy_first = df_22['signal'].value_counts(normalize=True)[1] * 100
    percentagetime_hold_first = df_22['signal'].value_counts(normalize=True, dropna=False).get("missing_key", 0) * 100
    percentagetime_sell_first = df_22['signal'].value_counts(normalize=True)[-1] * 100

    # NW 2STEPS LONG WITH THRESHOLD 
    return_2022_second = last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[-1] * 100 
    volatility_second = df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].std() * 100
    annualized_sharpe_ratio_second = ((df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].mean() - 0.02/365) / (df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].std()) * mt.sqrt(365)) 
    accuracy_strategy_second = df_22['accuracy strategy with threshold'].mean() * 100
    accuracy_signals_second = df_22['accuracy signals'].mean() * 100 
    better_arms_second = df_22['BETTER ARMS with threshold'].mean() * 100 
    difference_from_ATH_second = (df_22['NW 2STEPS LONG WITH THRESHOLD'].iloc[-1] / df_22['NW 2STEPS LONG WITH THRESHOLD'].max() - 1) * 100
    maximum_daily_gain_second = df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].max() * 100
    maximum_daily_loss_second = df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD RATIO'].min() * 100
    mean_monthly_return_second = df_22_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_second = df_22_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_second = df_22_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD LAST HOUR AND DAY RATIO'].min() * 100
    percentagetime_in_second = df_22['in/out 2STEPS LONG WITH THRESHOLD'].mean() * 100
    percentagetime_buy_second = df_22['buy/hold/sell'].value_counts(normalize=True)[1] * 100
    percentagetime_hold_second = df_22['buy/hold/sell'].value_counts(normalize=True, dropna=False).get(0, 0) * 100
    percentagetime_sell_second = df_22['buy/hold/sell'].value_counts(normalize=True)[-1] * 100

    # NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL
    return_2022_third = last_day_of_the_year['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY OF THE YEAR RATIO'].iloc[-1] * 100
    volatility_third = df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].std() * 100
    annualized_sharpe_ratio_third = ((df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].mean() - 0.02/365) / (df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].std()) * mt.sqrt(365)) 
    accuracy_strategy_third = df_22['accuracy strategy with threshold and selective sell'].mean() * 100
    accuracy_signals_third = df_22['accuracy signals'].mean() * 100
    better_arms_third = df_22['BETTER ARMS  with threshold and ss'].mean() * 100
    difference_from_ATH_third = (df_22['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].iloc[-1] / df_22['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].max() - 1) * 100
    maximum_daily_gain_third = df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].max() * 100
    maximum_daily_loss_third = df_22_last_hour['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL RATIO'].min() * 100
    mean_monthly_return_third = df_22_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_third = df_22_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_third = df_22_last_hour_last_day['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL LAST HOUR AND DAY RATIO'].min() * 100
    percentagetime_in_third = df_22['in/out 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'].mean() * 100
    percentagetime_buy_third = df_22['buy/hold/sell with selective sell'].value_counts(normalize=True)[1] * 100
    percentagetime_hold_third = df_22['buy/hold/sell with selective sell'].value_counts(normalize=True)[0] * 100
    percentagetime_sell_third = df_22['buy/hold/sell with selective sell'].value_counts(normalize=True)[-1] * 100

    #BTC HOLD
    return_2022_forth = last_day_of_the_year['btc hold RATIO'].iloc[0] * 100
    volatility_forth = df_22_last_hour['btc hold RATIO'].std() * 100
    annualized_sharpe_ratio_forth = ((df_22_last_hour['btc hold RATIO'].mean() - 0.02/365) / (df_22_last_hour['btc hold RATIO'].std()) * mt.sqrt(365)) * 100
    accuracy_strategy_forth = np.nan
    accuracy_signals_forth = np.nan
    better_arms_forth = np.nan
    difference_from_ATH_forth = (df_22['btc hold'].iloc[-1] / df_22['btc hold'].max() - 1) * 100
    maximum_daily_gain_forth = df_22_last_hour['btc hold RATIO'].max() * 100
    maximum_daily_loss_forth = df_22_last_hour['btc hold RATIO'].min() * 100
    mean_monthly_return_forth = df_22_last_hour_last_day['btc hold LAST HOUR AND DAY RATIO'].mean() * 100
    best_monthly_return_forth = df_22_last_hour_last_day['btc hold LAST HOUR AND DAY RATIO'].max() * 100
    worst_monthly_return_forth = df_22_last_hour_last_day['btc hold LAST HOUR AND DAY RATIO'].min() * 100
    percentagetime_in_forth = np.nan
    percentagetime_buy_forth = np.nan
    percentagetime_hold_forth = np.nan
    percentagetime_sell_forth = np.nan

    index_column = ['return','volatility', 'annualized sharpe ratio', 'ACCURACY strategy', 'accuracy signals', 'BETTER ARMS',
                'difference from ATH', 'maximum daily gain', 'maximum daily loss', 'mean monthly return',
                'best monthly return', 'worst month return', '%time in', '% buy',
                '% hold', '% sell']

    first_data = [return_2022_first, volatility_first, annualized_sharpe_ratio_first, accuracy_strategy_first, accuracy_signals_first, better_arms_first, difference_from_ATH_first, maximum_daily_gain_first, 
    maximum_daily_loss_first, mean_monthly_return_first, best_monthly_return_first, worst_monthly_return_first, percentagetime_in_first, percentagetime_buy_first, percentagetime_hold_first, percentagetime_sell_first]
 
    second_data = [return_2022_second,volatility_second,annualized_sharpe_ratio_second,accuracy_strategy_second,accuracy_signals_second,better_arms_second,difference_from_ATH_second,maximum_daily_gain_second,
    maximum_daily_loss_second,mean_monthly_return_second,best_monthly_return_second,worst_monthly_return_second,percentagetime_in_second,percentagetime_buy_second,percentagetime_hold_second,
    percentagetime_sell_second]

    third_data = [return_2022_third,volatility_third,annualized_sharpe_ratio_third,accuracy_strategy_third,accuracy_signals_third,better_arms_third,difference_from_ATH_third,maximum_daily_gain_third,
    maximum_daily_loss_third,mean_monthly_return_third,best_monthly_return_third,worst_monthly_return_third,percentagetime_in_third,percentagetime_buy_third,
    percentagetime_hold_third,percentagetime_sell_third]

    forth_data = [return_2022_forth,volatility_forth,annualized_sharpe_ratio_forth,accuracy_strategy_forth,accuracy_signals_forth,better_arms_forth,difference_from_ATH_forth,maximum_daily_gain_forth,maximum_daily_loss_forth,
    mean_monthly_return_forth,best_monthly_return_forth,worst_monthly_return_forth,percentagetime_in_forth,percentagetime_buy_forth,percentagetime_hold_forth,
    percentagetime_sell_forth]

    important_scores = pd.DataFrame(columns=['NW 2STEPS LONG NO THRESHOLD'], index=index_column)
    important_scores['NW 2STEPS LONG NO THRESHOLD'] = first_data
    important_scores['NW 2STEPS LONG WITH THRESHOLD'] = second_data
    important_scores['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = third_data
    important_scores['BTC HOLD'] = forth_data

    return important_scores

def first_strategy(important_scores_21,important_scores_22):

    TWO_STEPS_LONG_NO_THRESHOLD = pd.DataFrame()
    TWO_STEPS_LONG_NO_THRESHOLD['2021'] = important_scores_df_21['NW 2STEPS LONG NO THRESHOLD']
    TWO_STEPS_LONG_NO_THRESHOLD['2022'] = important_scores_df_22['NW 2STEPS LONG NO THRESHOLD']
    TWO_STEPS_LONG_NO_THRESHOLD['2021-2022'] = important_scores_df['NW 2STEPS LONG NO THRESHOLD']
    TWO_STEPS_LONG_NO_THRESHOLD = (TWO_STEPS_LONG_NO_THRESHOLD.style
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-weight', 'bold')]}]))
    return TWO_STEPS_LONG_NO_THRESHOLD

def second_strategy(important_scores_21,important_scores_22):
    
    TWO_STEPS_LONG_WITH_THRESHOLD = pd.DataFrame()
    TWO_STEPS_LONG_WITH_THRESHOLD['2021'] = important_scores_df_21['NW 2STEPS LONG WITH THRESHOLD']
    TWO_STEPS_LONG_WITH_THRESHOLD['2022'] = important_scores_df_22['NW 2STEPS LONG WITH THRESHOLD']
    TWO_STEPS_LONG_WITH_THRESHOLD['2021-2022'] = important_scores_df['NW 2STEPS LONG WITH THRESHOLD']
    TWO_STEPS_LONG_WITH_THRESHOLD = (TWO_STEPS_LONG_WITH_THRESHOLD.style
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-weight', 'bold')]}]))
    return TWO_STEPS_LONG_WITH_THRESHOLD

def third_strategy(important_scores_21,important_scores_22):

    TWO_STEPS_LONG_WITH_THRESHOLD_and_SS = pd.DataFrame()
    TWO_STEPS_LONG_WITH_THRESHOLD_and_SS['2021'] = important_scores_df_21['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL']
    TWO_STEPS_LONG_WITH_THRESHOLD_and_SS['2022'] = important_scores_df_22['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL']
    TWO_STEPS_LONG_WITH_THRESHOLD_and_SS['2021-2022'] = important_scores_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL']
    TWO_STEPS_LONG_WITH_THRESHOLD_and_SS = (TWO_STEPS_LONG_WITH_THRESHOLD_and_SS.style
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-weight', 'bold')]}]))
                                    
    return TWO_STEPS_LONG_WITH_THRESHOLD_and_SS


def is_correct_format(df):
    # Check if the DataFrame has the expected columns
    expected_columns = [
       'StartTime','Price Open','Price Close','Prediction'
    ]
    return all(column in df.columns for column in expected_columns)


def format_dataframe_values(df):
    formatted_df = df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:.2f}")
    return formatted_df


if uploaded_file:
    st.markdown(
    f'<div style="color:#3dfd9f">Uploaded file: <strong style="color:#ffffff">{uploaded_file.name}</strong></div>',
    unsafe_allow_html=True,
    )

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
            initial_df = pd.read_csv(uploaded_file, encoding='utf-8')
            if not is_correct_format(initial_df):
            # Reset the uploaded file to the start
                uploaded_file.seek(0)
                initial_df = pd.read_csv(uploaded_file, encoding='utf-8',header=None)
                # Convert the first column values to text
                csv_text = '\n'.join(initial_df.iloc[:, 0].tolist())
                # Create a new DataFrame from the text
                initial_df = pd.read_csv(StringIO(csv_text))
    elif uploaded_file.name.endswith(('.xlsx', '.xlsm')):
        initial_df = pd.read_excel(uploaded_file, engine='openpyxl')
        if not is_correct_format(initial_df):
            # Reset the uploaded file to the start
                uploaded_file.seek(0)
                initial_df = pd.read_excel(uploaded_file, engine='openpyxl',header=None)
                # Convert the first column values to text
                csv_text = '\n'.join(initial_df.iloc[:, 0].tolist())
                # Create a new DataFrame from the text
                initial_df = pd.read_csv(StringIO(csv_text))
    else:
        st.write("Unsupported file type. Please upload a CSV, XLSX, or XLSM file.")
    initial_df = initial_df[['StartTime','Price Open','Price Close','Prediction']]
    initial_df['StartTime'] = pd.to_datetime(initial_df['StartTime'])
    mask = (initial_df['StartTime'] >= '2021-01-01 00:00:00') & (initial_df['StartTime'] <= '2022-12-31 23:00:00')
    df_start = initial_df.copy()
    initial_df = initial_df.loc[mask]
    initial_df = initial_df.reset_index(drop=True)
    
    df_thresholds = threshold_summary([0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02],df_start)
    df_thresholds = format_dataframe_values(df_thresholds)
    main_df = automatize(initial_df['StartTime'], initial_df['Price Open'], initial_df['Price Close'], initial_df['Price Close'], threshold_decimal)
    
    last_hour = last_hour_df(main_df)
    last_hour_day = last_hour_and_day_df(main_df)
    last_day_of_the_year = last_day_of_the_year_last_hour(main_df)
 
    df_21 = filter_2021_df(main_df)
    df_22 = filter_2022_df(main_df)
    df_21_last_hour = filter_2021_last_hour_df(last_hour)
    df_22_last_hour = filter_2022_last_hour_df(last_hour)
    df_21_last_hour_last_day = filter_2021_last_hour_last_day(last_hour_day)
    df_22_last_hour_last_day = filter_2022_last_hour_last_day(last_hour_day)
    
    return_volatility_df = return_volatility(last_hour)

    important_scores_df = important_scores(main_df,last_hour,last_day_of_the_year)
    important_scores_df_21 = important_scores_21(main_df,last_hour,last_day_of_the_year)
    important_scores_df_22 = important_scores_22(main_df,last_hour,last_day_of_the_year)

    important_scores_df['NW 2STEPS LONG NO THRESHOLD'] = format_percentage(important_scores_df['NW 2STEPS LONG NO THRESHOLD'])
    important_scores_df['NW 2STEPS LONG WITH THRESHOLD'] = format_percentage(important_scores_df['NW 2STEPS LONG WITH THRESHOLD'])
    important_scores_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = format_percentage(important_scores_df['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'])
    important_scores_df['BTC HOLD'] = format_percentage(important_scores_df['BTC HOLD'])

    important_scores_df_21['NW 2STEPS LONG NO THRESHOLD'] = format_percentage(important_scores_df_21['NW 2STEPS LONG NO THRESHOLD'])
    important_scores_df_21['NW 2STEPS LONG WITH THRESHOLD'] = format_percentage(important_scores_df_21['NW 2STEPS LONG WITH THRESHOLD'])
    important_scores_df_21['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = format_percentage(important_scores_df_21['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'])
    important_scores_df_21['BTC HOLD'] = format_percentage(important_scores_df_21['BTC HOLD'])

    important_scores_df_22['NW 2STEPS LONG NO THRESHOLD'] = format_percentage(important_scores_df_22['NW 2STEPS LONG NO THRESHOLD'])
    important_scores_df_22['NW 2STEPS LONG WITH THRESHOLD'] = format_percentage(important_scores_df_22['NW 2STEPS LONG WITH THRESHOLD'])
    important_scores_df_22['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'] = format_percentage(important_scores_df_22['NW 2STEPS LONG WITH THRESHOLD AND SELECTIVE SELL'])
    important_scores_df_22['BTC HOLD'] = format_percentage(important_scores_df_22['BTC HOLD'])

    first_strategy_df = first_strategy(important_scores_df_21,important_scores_df_22)
    second_strategy_df = second_strategy(important_scores_df_21,important_scores_df_22)
    third_strategy_df = third_strategy(important_scores_df_21,important_scores_df_22)
    fig1 = networth_evolution(last_hour)
    fig2 = networth_evolution_each_day(last_hour)

table_style = """
<style>
    .table_full_width {
        width: 100%;
        background-color: #0F1117;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    }
    .table_full_width td,
    .table_full_width th {
        text-align: center;
        border: 1px solid #2a2d34;
        padding: 2px;
        font-size: 15px;
        line-height: 1.2
        color: white;
    }
    .table_full_width thead th {
        background-color: #262730;
        color: white;
        font-weight: bold;
        border-bottom: 2px solid #2a2d34;
    }
    .table_full_width tbody tr {
        background-color: #0F1117;
    }
    .table_full_width tbody tr:hover {
        background-color: #00C07F;
    }
</style>
"""

if submit_button:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #3dfd9f;font-size: 20px;'>Threshold Summary</h3>", unsafe_allow_html=True)
    
    table1_html = df_thresholds.to_html(classes="table_full_width")
    table2_html = first_strategy_df.to_html(classes="table_full_width")
    table3_html = second_strategy_df.to_html(classes="table_full_width")
    table4_html = third_strategy_df.to_html(classes="table_full_width")
    table5_html = return_volatility_df.to_html(classes="table_full_width")
    st.write(f'{table_style}{table1_html}', unsafe_allow_html=True)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

    col1, col2, col3 = st.columns(3)

    col1.markdown("<h3 style='color: #3dfd9f;font-size: 20px;'>Trading Strategy</h3>", unsafe_allow_html=True)
    col2.markdown("<h3 style='color: #3dfd9f;font-size: 20px;'>Low Exposure Strategy</h3>", unsafe_allow_html=True)
    col3.markdown("<h3 style='color: #3dfd9f;font-size: 20px;'>High Exposure Strategy</h3>", unsafe_allow_html=True)



    col1.write(f'{table_style}{table2_html}', unsafe_allow_html=True)
    col2.write(f'{table_style}{table3_html}', unsafe_allow_html=True)
    col3.write(f'{table_style}{table4_html}', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #3dfd9f;font-size: 20px;'>Return and Volatility</h3>", unsafe_allow_html=True)
    st.write(f'{table_style}{table5_html}', unsafe_allow_html=True)
