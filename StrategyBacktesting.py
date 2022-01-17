import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import altair as alt
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('EMA Crossover Vs. Supertrend Trading Strategies')
input_col1, input_col2 = st.columns(2)
initial_capital = input_col1.number_input(label="Initial Capital", step=1, format="%.i")
position_size = input_col1.number_input("Position Size (% of Capital Per Trade)", 1, 100, 1, 1)
symbol = input_col1.selectbox('Instrument (Blue Chip Stocks, NSE)', ('RELIANCE', 'TCS', 'HINDUNILVR', 'HDFCBANK', 'HDFC', 'INFY',
                                                         'KOTAKBANK', 'BHARTIARTL', 'ITC', 'ICICIBANK', 'SBIN',
                                                         'ASIANPAINT', 'BAJFINANCE', 'MARUTI', 'HCLTECH', 'LT',
                                                         'WIPRO', 'AXISBANK', 'ONGC', 'SUNPHARMA'))
testing_period = input_col1.selectbox('Testing Period (in Days)', (1000, 2000, 3000, 4000))
ema_a_length = input_col2.number_input(label="Period For EMA 1 :", step=1, format="%.i")
ema_b_length = input_col2.number_input(label="Period For EMA 2 :", step=1, format="%.i")
super_period = input_col2.number_input(label="Supertrend ATR Period :", step=1, format="%.i")
super_multiplier = input_col2.number_input(label="Supertrend ATR Multiplier", step=1., format="%.1f")
_, _, _, col, _, _, _ = st.columns([2] * 6 + [1])
apply = col.button('Run a Backtest')
if apply:
    ohlc_df = yf.download(symbol + '.NS', progress=False)
    ohlc_df.drop(
        index=ohlc_df.index[:(len(ohlc_df.index) - (testing_period + max(ema_a_length, ema_b_length, super_period)))],
        columns=['Adj Close', 'Volume'], axis=0, inplace=True)
    ohlc_df.to_pickle(symbol + '.pkl')
    current_price = ohlc_df.Close.iloc[-1]

    ema_df = ohlc_df.copy()
    ema_df['EMA_A'] = ta.ema(ema_df['Close'], length=ema_a_length)
    ema_df['EMA_B'] = ta.ema(ema_df['Close'], length=ema_b_length)

    ema_df.drop(index=ema_df.index[:(len(ema_df.index) - (testing_period + 1))], axis=0, inplace=True)

    ema_df.loc[(ema_df['EMA_A'] < ema_df['EMA_B']), 'EMASign'] = -1
    ema_df.loc[(ema_df['EMA_A'] == ema_df['EMA_B']), 'EMASign'] = 0
    ema_df.loc[(ema_df['EMA_A'] > ema_df['EMA_B']), 'EMASign'] = 1

    ema_df.loc[(ema_df['EMASign'] == 1) & (ema_df['EMASign'].shift(1) < ema_df['EMASign']), 'Exit Strategy'] = 'LONG'
    ema_df.loc[
        (ema_df['EMASign'] == -1) & (ema_df['EMASign'].shift(1) > ema_df['EMASign']), 'Exit Strategy'] = 'EMA Cross-Down'

    ema_df.loc[(ema_df['Exit Strategy'] == 'LONG'), 'SL'] = ema_df['Close'] - ema_df['Close'] * 0.05

    for (row, rs) in ema_df.iterrows():
        ema_df.loc[(ema_df['Exit Strategy'] != 'LONG') & (ema_df['EMASign'] == 1), 'SL'] = ema_df['SL'].shift(1)

    ema_df.loc[(ema_df['Low'] <= ema_df['SL']), 'Exit Strategy'] = 'SL Hit'

    ema_df = ema_df.loc[(ema_df['Exit Strategy'] == 'LONG') | (ema_df['Exit Strategy'] == 'EMA Cross-Down') |
                        (ema_df['Exit Strategy'] == 'SL Hit') | (ema_df['Close'] == current_price)]

    ema_df = ema_df.loc[(ema_df['Exit Strategy'] == 'LONG') | (ema_df['Exit Strategy'].shift(1) == 'LONG')]

    ema_df.loc[(ema_df['Exit Strategy'] != 'LONG'), 'Status'] = 'OPEN'
    ema_df.loc[(ema_df['Exit Strategy'] == 'EMA Cross-Down') | (ema_df['Exit Strategy'] == 'SL Hit'), 'Status'] = 'CLOSED'

    ema_df.loc[(ema_df['Exit Strategy'] == 'EMA Cross-Down') | (ema_df['Status'] == 'OPEN'),
               'PnL Percentage'] = ((ema_df['Close'] - ema_df['Close'].shift(1)) / (ema_df['Close'].shift(1))) * 100
    ema_df.loc[(ema_df['Exit Strategy'] == 'SL Hit'),
               'PnL Percentage'] = ((ema_df['SL'] - ema_df['Close'].shift(1)) / (ema_df['Close'].shift(1))) * 100

    ema_df.loc[(ema_df['PnL Percentage'] < -5), 'PnL Percentage'] = -5
    ema_df.loc[(ema_df['PnL Percentage'] <= -5), 'Exit Strategy'] = 'SL Hit'

    ema_df.drop(columns=['Open', 'High', 'Low', 'EMA_A', 'EMA_B', 'EMASign'], axis=0, inplace=True)

    ema_df['Direction'] = 'LONG'
    ema_df['DateCopy'] = ema_df.index
    ema_df['Start Date'] = ema_df['DateCopy'].shift(1)
    ema_df.loc[(ema_df['Status'] == 'CLOSED'), 'End Date'] = ema_df['DateCopy']

    ema_df['Entry Price'] = ema_df['Close'].shift(1)
    ema_df.loc[(ema_df['Exit Strategy'] == 'EMA Cross-Down'), 'Exit Price'] = ema_df['Close']
    ema_df.loc[(ema_df['Exit Strategy'] == 'SL Hit'), 'Exit Price'] = ema_df['Close'].shift(1) - ema_df['Close'].shift(
        1) * 0.05

    current_date = ema_df.DateCopy.iloc[-1]

    ema_df = ema_df.loc[(ema_df['Status'] == 'CLOSED') | (ema_df['Status'] == 'OPEN')]
    ema_df = ema_df.reset_index()

    ema_df['Account'] = 0
    ema_df['Position Size'] = 0
    ema_df['Profit/Loss'] = 0

    ema_df.iloc[0] = ['1900-01-01', 0, 0, 0, 0, 0, 0, '1900-01-01', '1900-01-01', '1900-01-01', 0, 0, initial_capital, 0, 0]

    for (row, rs) in ema_df.iterrows():
        ema_df.loc[ema_df.index != 0, 'Position Size'] = ema_df['Account'].shift(1) * (position_size / 100)
        ema_df.loc[ema_df.index != 0, 'Profit/Loss'] = ema_df['Position Size'] * (ema_df['PnL Percentage'] / 100)
        ema_df.loc[ema_df.index != 0, 'Account'] = ema_df['Account'].shift(1) + ema_df['Profit/Loss']

    ema_df.drop(columns=['Date', 'Close', 'DateCopy', 'SL'], axis=0, inplace=True)
    ema_df.drop(index=ema_df.index[0], axis=0, inplace=True)
    ema_df = ema_df[['Direction', 'Start Date', 'Entry Price', 'Status', 'End Date', 'Exit Price',
                     'Exit Strategy', 'Position Size', 'PnL Percentage', 'Profit/Loss', 'Account']]

    ema_df['WRow'] = 0
    ema_df['LRow'] = 0
    ema_df.loc[ema_df['PnL Percentage'] > 0, 'WRow'] = 1
    ema_df.loc[ema_df['PnL Percentage'] < 0, 'LRow'] = 1

    for (row, rs) in ema_df.iterrows():
        ema_df.loc[(ema_df['PnL Percentage'] > 0) & (ema_df.index != 1), 'WRow'] = 1 + ema_df['WRow'].shift(1)
        ema_df.loc[(ema_df['PnL Percentage'] < 0) & (ema_df.index != 1), 'LRow'] = 1 + ema_df['LRow'].shift(1)

    ema_df['WRow'] = ema_df['WRow'].astype(int)
    ema_df['LRow'] = ema_df['LRow'].astype(int)
    ema_df = ema_df.round(decimals=2)

    instrument1 = symbol
    timeframe1 = 'Daily'
    start_date1 = ema_df['Start Date'].iloc[0].strftime('%Y-%m-%d')
    end_date1 = current_date.strftime('%Y-%m-%d')
    net_profit1 = ema_df['Account'].iloc[-1] - initial_capital
    net_percentage1 = (net_profit1 / initial_capital) * 100
    gross_profit1 = (sum(ema_df['Profit/Loss'].to_numpy().clip(min=0)) / initial_capital) * 100
    gross_loss1 = (sum(ema_df['Profit/Loss'].to_numpy().clip(max=0)) / initial_capital) * 100
    buyhold_return1 = ((current_price - ema_df['Entry Price'].iloc[0]) / ema_df['Entry Price'].iloc[0]) * 100
    no_of_trades1 = len(ema_df)
    no_of_wins1 = len(ema_df.loc[ema_df['Profit/Loss'] >= 0])
    no_of_losses1 = len(ema_df.loc[ema_df['Profit/Loss'] < 0])
    winning_rate1 = (no_of_wins1 / no_of_trades1) * 100
    avg_profit1 = net_percentage1 / no_of_trades1
    profit_factor1 = sum(ema_df['Profit/Loss'].to_numpy().clip(min=0)) / (
                sum(ema_df['Profit/Loss'].to_numpy().clip(max=0)) * (-1))
    max_drawdown1 = min(ema_df['Account']) - max(ema_df['Account'])
    trough1 = min(ema_df['Account'])
    if trough1 > initial_capital:
        trough1 = initial_capital
    peak1 = max(ema_df['Account'])
    if peak1 < initial_capital:
        peak1 = initial_capital
    largest_win1 = max(ema_df['PnL Percentage'])
    largest_loss1 = min(ema_df['PnL Percentage'])
    wins_row1 = max(ema_df['WRow'])
    loss_row1 = max(ema_df['LRow'])

    super_df = ohlc_df.copy()
    super_df.ta.supertrend(length=super_period, multiplier=super_multiplier, append=True)

    super_df['SUPERSign'] = super_df['SUPERTd_' + str(super_period) + '_' + str(super_multiplier)]

    super_df.drop(columns=['SUPERT_' + str(super_period) + '_' + str(super_multiplier),
                           'SUPERTd_' + str(super_period) + '_' + str(super_multiplier),
                           'SUPERTl_' + str(super_period) + '_' + str(super_multiplier),
                           'SUPERTs_' + str(super_period) + '_' + str(super_multiplier)], axis=0, inplace=True)

    super_df.drop(index=super_df.index[:(len(super_df.index) - (testing_period + 1))], axis=0, inplace=True)

    super_df.loc[
        (super_df['SUPERSign'] == 1) & (super_df['SUPERSign'].shift(1) < super_df['SUPERSign']), 'Exit Strategy'] = 'LONG'
    super_df.loc[(super_df['SUPERSign'] == -1) & (
                super_df['SUPERSign'].shift(1) > super_df['SUPERSign']), 'Exit Strategy'] = 'Supertrend SHORT'

    super_df = super_df.loc[(super_df['Exit Strategy'] == 'LONG') | (super_df['Exit Strategy'] == 'Supertrend SHORT') | (
                super_df['Close'] == current_price)]

    super_df = super_df.loc[(super_df['Exit Strategy'] == 'LONG') | (super_df['Exit Strategy'].shift(1) == 'LONG')]

    super_df.loc[(super_df['Exit Strategy'] != 'LONG'), 'Status'] = 'OPEN'
    super_df.loc[(super_df['Exit Strategy'] == 'Supertrend SHORT') | (super_df['Exit Strategy'] == 'SL Hit'),
                 'Status'] = 'CLOSED'

    super_df.loc[(super_df['Exit Strategy'] == 'Supertrend SHORT') | (super_df['Status'] == 'OPEN'),
                 'PnL Percentage'] = ((super_df['Close'] - super_df['Close'].shift(1)) / (super_df['Close'].shift(1))) * 100

    super_df.drop(columns=['Open', 'High', 'Low', 'SUPERSign'], axis=0, inplace=True)

    super_df['Direction'] = 'LONG'
    super_df['DateCopy'] = super_df.index
    super_df['Start Date'] = super_df['DateCopy'].shift(1)
    super_df.loc[(super_df['Status'] == 'CLOSED'), 'End Date'] = super_df['DateCopy']

    super_df['Entry Price'] = super_df['Close'].shift(1)
    super_df.loc[(super_df['Exit Strategy'] == 'Supertrend SHORT'), 'Exit Price'] = super_df['Close']

    current_date = super_df.DateCopy.iloc[-1]

    super_df = super_df.loc[(super_df['Status'] == 'CLOSED') | (super_df['Status'] == 'OPEN')]

    super_df = super_df.reset_index()

    super_df['Account'] = 0
    super_df['Position Size'] = 0
    super_df['Profit/Loss'] = 0

    super_df.iloc[0] = ['1900-01-01', 0, 0, 0, 0, 0, '1900-01-01', '1900-01-01', '1900-01-01', 0, 0, initial_capital, 0, 0]

    for (row, rs) in super_df.iterrows():
        super_df.loc[super_df.index != 0, 'Position Size'] = super_df['Account'].shift(1) * (position_size / 100)
        super_df.loc[super_df.index != 0, 'Profit/Loss'] = super_df['Position Size'] * (super_df['PnL Percentage'] / 100)
        super_df.loc[super_df.index != 0, 'Account'] = super_df['Account'].shift(1) + super_df['Profit/Loss']

    super_df.drop(columns=['Date', 'Close', 'DateCopy'], axis=0, inplace=True)

    super_df.drop(index=super_df.index[0], axis=0, inplace=True)
    super_df = super_df[['Direction', 'Start Date', 'Entry Price', 'Status', 'End Date', 'Exit Price',
                         'Exit Strategy', 'Position Size', 'PnL Percentage', 'Profit/Loss', 'Account']]

    super_df['WRow'] = 0
    super_df['LRow'] = 0
    super_df.loc[super_df['PnL Percentage'] > 0, 'WRow'] = 1
    super_df.loc[super_df['PnL Percentage'] < 0, 'LRow'] = 1

    for (row, rs) in super_df.iterrows():
        super_df.loc[(super_df['PnL Percentage'] > 0) & (super_df.index != 1), 'WRow'] = 1 + super_df['WRow'].shift(1)
        super_df.loc[(super_df['PnL Percentage'] < 0) & (super_df.index != 1), 'LRow'] = 1 + super_df['LRow'].shift(1)

    super_df['WRow'] = super_df['WRow'].astype(int)
    super_df['LRow'] = super_df['LRow'].astype(int)
    super_df = super_df.round(decimals=2)

    instrument2 = symbol
    timeframe2 = 'Daily'
    start_date2 = super_df['Start Date'].iloc[0].strftime('%Y-%m-%d')
    end_date2 = current_date.strftime('%Y-%m-%d')
    net_profit2 = super_df['Account'].iloc[-1] - initial_capital
    net_percentage2 = (net_profit2 / initial_capital) * 100
    gross_profit2 = (sum(super_df['Profit/Loss'].to_numpy().clip(min=0)) / initial_capital) * 100
    gross_loss2 = (sum(super_df['Profit/Loss'].to_numpy().clip(max=0)) / initial_capital) * 100
    buyhold_return2 = ((current_price - super_df['Entry Price'].iloc[0]) / super_df['Entry Price'].iloc[0]) * 100
    no_of_trades2 = len(super_df)
    no_of_wins2 = len(super_df.loc[super_df['Profit/Loss'] >= 0])
    no_of_losses2 = len(super_df.loc[super_df['Profit/Loss'] < 0])
    winning_rate2 = (no_of_wins2 / no_of_trades2) * 100
    avg_profit2 = net_percentage2 / no_of_trades2
    profit_factor2 = sum(super_df['Profit/Loss'].to_numpy().clip(min=0)) / (
                sum(super_df['Profit/Loss'].to_numpy().clip(max=0)) * (-1))
    max_drawdown2 = min(super_df['Account']) - max(super_df['Account'])
    trough2 = min(super_df['Account'])
    if trough2 > initial_capital:
        trough2 = initial_capital
    peak2 = max(super_df['Account'])
    if peak2 < initial_capital:
        peak2 = initial_capital
    largest_win2 = max(super_df['PnL Percentage'])
    largest_loss2 = min(super_df['PnL Percentage'])
    wins_row2 = max(super_df['WRow'])
    loss_row2 = max(super_df['LRow'])

    data = {'Parameter': ['Instrument', 'Timeframe', 'Start Date', 'End Date', 'Net Profit, RoI (INR)',
                          'Net Profit, RoI %', 'Gross Profit', 'Gross Loss', 'Buy and Hold Returns',
                          'Total No.of Trades', 'No.of Winning Trades', 'No.of Losing Trades', 'Strategy Winning Rate',
                          'Average Profit Per Trade', 'Profit Factor', 'Max Drawdown (INR)',
                          'Account Low (INR)', 'Account Peak (INR)', 'Largest Winning Trade', 'Largest Losing Trade',
                          'Most Wins in a Row', 'Most Losses in a Row'],
            'EMA Crossover Strategy': [instrument1, timeframe1, start_date1, end_date1, str(net_profit1.round(decimals=2)),
                              str(net_percentage1.round(decimals=2)) + ' %',
                              str(gross_profit1.round(decimals=2)) + ' %', str(gross_loss1.round(decimals=2)) + ' %',
                              str(buyhold_return1.round(decimals=2)) + ' %', no_of_trades1, no_of_wins1, no_of_losses1,
                              str(round(winning_rate1, 2)) + ' %', str(round(avg_profit1, 2)) + ' %',
                              str(profit_factor1.round(decimals=2)), str(round(max_drawdown1, 2)), trough1, peak1,
                              str(round(largest_win1, 2)) + ' %', str(round(largest_loss1, 2)) + ' %',
                              wins_row1, loss_row1],
            'Supertrend Strategy': [instrument2, timeframe2, start_date2, end_date2, str(net_profit2.round(decimals=2)),
                           str(net_percentage2.round(decimals=2)) + ' %', str(gross_profit2.round(decimals=2)) + ' %',
                           str(gross_loss2.round(decimals=2)) + ' %', str(buyhold_return2.round(decimals=2)) + ' %',
                           no_of_trades2, no_of_wins2, no_of_losses2, str(round(winning_rate2, 2)) + ' %',
                           str(round(avg_profit2, 2)) + ' %', str(profit_factor2.round(decimals=2)),
                           str(round(max_drawdown2, 2)), trough2, peak2, str(round(largest_win2, 2)) + ' %',
                           str(round(largest_loss2, 2)) + ' %', wins_row2, loss_row2]}

    summary_df = pd.DataFrame(data)
    st.header("Performance Summary")
    str_summary_df = summary_df.astype(str)
    styler = str_summary_df.style.hide_index().bar(align="mid")
    st.write(styler.to_html(), unsafe_allow_html=True)
    st.header('List of Trades (EMA Crossover)')
    ema_df.drop(columns=['WRow', 'LRow'], axis=0, inplace=True)
    str_ema_df = ema_df.astype(str)
    st.dataframe(str_ema_df)
    st.header('List of Trades (Supertrend)')
    super_df.drop(columns=['WRow', 'LRow'], axis=0, inplace=True)
    str_super_df = super_df.astype(str)
    st.dataframe(str_super_df)

    # Line chart
    st.header("Equity Curve (EMA Crossover)")
    chart_col5, chart_col6 = st.columns(2)

    chart_df_5 = ema_df.copy()
    chart_df_5.drop(columns=['Direction', 'Entry Price', 'Status', 'End Date', 'Exit Price', 'Exit Strategy',
                             'Position Size', 'PnL Percentage', 'Profit/Loss'], axis=0, inplace=True)
    chart_df_5 = chart_df_5.rename(columns={"Start Date": "Period"})
    chart = alt.Chart(chart_df_5).mark_line().encode(x='Period:T', y=alt.X('Account:Q', scale=alt.Scale(zero=False))). \
        properties(width=650, height=500, title="Equity Curve (EMA Crossover)").interactive()
    st.altair_chart(chart, use_container_width=True)

    st.header("Equity Curve (Supertrend)")
    chart_df_6 = super_df.copy()
    chart_df_6.drop(
        columns=['Direction', 'Entry Price', 'Status', 'End Date', 'Exit Price', 'Exit Strategy', 'Position Size',
                 'PnL Percentage', 'Profit/Loss'], axis=0, inplace=True)
    chart_df_6 = chart_df_6.rename(columns={"Start Date": "Period"})
    chart = alt.Chart(chart_df_6).mark_line().encode(x='Period:T', y=alt.X('Account:Q', scale=alt.Scale(zero=False))). \
        properties(width=650, height=500, title="Equity Curve (Supertrend)").interactive()
    st.altair_chart(chart, use_container_width=True)

    #pie chart
    st.header("Strategy Winning Rate Comparison")

    chart_col3, chart_col4 = st.columns(2)

    chart_data_3 = {'Parameter': ['Wins', 'Losses'],
                    'Value': [no_of_wins1, no_of_losses1]}
    chart_df_3 = pd.DataFrame(chart_data_3)
    chart_df_3 = chart_df_3.round(decimals=2)
    labels = 'Win', 'Losses'
    pie_chart_df1 = pd.DataFrame(chart_df_3)
    sizes = [no_of_wins1, no_of_losses1]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=0)
    ax1.axis('equal')
    chart_col3.text('Winning Rate (EMA Crossover)')
    chart_col3.pyplot(fig1)
    styler1 = pie_chart_df1.style.hide_index().bar(align="mid")
    chart_col3.write(styler1.to_html(), unsafe_allow_html=True)

    chart_data_4 = {'Parameter': ['Wins', 'Losses'],
                    'Value': [no_of_wins2, no_of_losses2]}
    chart_df_4 = pd.DataFrame(chart_data_4)
    chart_df_4 = chart_df_4.round(decimals=2)
    labels = 'Win', 'Losses'
    pie_chart_df2 = pd.DataFrame(chart_df_4)
    sizes = [no_of_wins2, no_of_losses2]
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=0)
    ax2.axis('equal')
    chart_col4.text('Winning Rate (Supertrend)')
    chart_col4.pyplot(fig2)
    styler2 = pie_chart_df2.style.hide_index().bar(align="mid")
    chart_col4.write(styler2.to_html(), unsafe_allow_html=True)

    # Area Chart
    st.header("Profit/Loss Percentage Over Trades")
    chart_col7, chart_col8 = st.columns(2)
    chart_df_7 = ema_df.copy()
    chart_df_7.drop(columns=['Direction', 'Entry Price', 'Status', 'End Date', 'Exit Price', 'Exit Strategy', 'Position Size',
                             'Account', 'Profit/Loss'], axis=0, inplace=True)
    chart_df_7 = chart_df_7.round(decimals=2)
    chart_data1 = pd.DataFrame(chart_df_7, columns=['PnL Percentage'])
    chart_col7.text("PnL Percentage Per Trade (EMA Crossover)")
    chart_col7.area_chart(chart_data1, use_container_width=True)

    chart_df_8 = ema_df.copy()
    chart_df_8.drop(columns=['Direction', 'Entry Price', 'Status', 'End Date', 'Exit Price', 'Exit Strategy', 'Position Size',
                             'Account', 'Profit/Loss'], axis=0, inplace=True)
    chart_df_8 = chart_df_7.round(decimals=2)
    chart_data2 = pd.DataFrame(chart_df_8, columns=['PnL Percentage'])
    chart_col8.text("PnL Percentage Per Trade (Supertrend)")
    chart_col8.area_chart(chart_data2, use_container_width=True)

    # Bar Chart
    st.header("RoI and Profit Factor Comparison")

    chart_col1, chart_col2 = st.columns(2)

    chart_data_1 = {'Strategy': ['EMA Crossover', 'Supertrend'],
                    'Gain on Account': [net_percentage1, net_percentage2]}
    chart_df_1 = pd.DataFrame(chart_data_1)
    chart_df_1 = chart_df_1.round(decimals=2)
    bar1 = alt.Chart(chart_df_1).mark_bar().encode(x='Strategy', y='Gain on Account'). \
        properties(width=200, height=600, title="Return on Investment").interactive()
    chart_col1.altair_chart(bar1)

    chart_data_2 = {'Strategy': ['EMA Crossover', 'Supertrend'],
                    'Profit Factor': [profit_factor1, profit_factor2]}
    chart_df_2 = pd.DataFrame(chart_data_2)
    chart_df_2 = chart_df_2.round(decimals=2)
    bar2 = alt.Chart(chart_df_2).mark_bar().encode(x='Strategy', y='Profit Factor'). \
        properties(width=200, height=600, title="Profit Factor").interactive()
    chart_col2.altair_chart(bar2)