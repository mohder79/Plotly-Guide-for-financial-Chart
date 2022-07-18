# Plotly Guide for Financial Chart

how to use plotly lib for Financial Chart


# import libraries  <br/>

```
import ccxt
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta

```



# **load exchange** <br/>

```

exchange = ccxt.binance({
    'options': {
        'adjustForTimeDifference': True,
    },

})

```
# **creat function for load data** <br/>
```
def fetch(symbol: str, timeframe: str, limit: int):
    print(f"Fetching {symbol} new bars for {datetime.now().isoformat()}")

    bars = exchange.fetch_ohlcv(
        symbol, timeframe=timeframe, limit=limit)  # fetch ohlcv
    df = pd.DataFrame(bars[:-1], columns=['timestamp',
                      'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index(pd.DatetimeIndex(df.timestamp))
    return df
```

# **set symbol for data function** <br/>
```
BTC = fetch('BTC/USDT', '1h', 1000)
```

# **Candlestick Chart** <br/>
```
fig = go.Figure()

fig.add_trace(go.Candlestick(x=BTC.index,
                             open=BTC['open'],
                             high=BTC['high'],
                             low=BTC['low'],
                             close=BTC['close'],
                             showlegend=False))
fig.show()
```
<br/>

![newplot (5)](https://user-images.githubusercontent.com/102425717/179375003-dca8d0b9-5af3-4dfa-87a6-4c09651374fe.png)<br/>

# **change Candlestick Chart color** <br/>

```
fig.add_trace(go.Candlestick(
    x=BTC.index,
    open=BTC['open'], high=BTC['high'],
    low=BTC['low'], close=BTC['close'],
    increasing_line_color='cyan', decreasing_line_color='gray'
))
```
![newplot (10)](https://user-images.githubusercontent.com/102425717/179378535-a0bdf45d-366c-4e01-b3bb-cbba948485c3.png)<br/>


# **removing rangeslider** <br/>
```
fig.update_layout(xaxis_rangeslider_visible=False)
```
# **hide Shadow** <br/>
```
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
```
# ** add moving averages to data frame and plot them** <br/>
```
BTC['sam15'] = ta.sma(BTC.close , 15)

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['sam15'],
                         opacity=0.7,
                         line=dict(color='Black', width=2),
                         name='sma'))
```
![newplot (6)](https://user-images.githubusercontent.com/102425717/179375190-afc97cc8-bb11-4835-bc9d-d3a901e3d0c6.png)<br/>

# ** change plot color with condition and set new dataframes** <br/>
```
BTC.loc[BTC['sam15'] >= BTC['close'], 'upper_sma15'] = BTC['sam15']
BTC.loc[BTC['sam15'] < BTC['close'], 'lower_sma15'] = BTC['sam15']

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['upper_sma15'],
                         opacity=1,
                         line=dict(color='Lime', width=4),
                         name='upper_sma'))

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['lower_sma15'],
                         opacity=1,
                         line=dict(color='Maroon', width=4),
                         name='lower_sma'))
```
![newplot (7)](https://user-images.githubusercontent.com/102425717/179375383-e0f7cb7c-c6eb-4b2a-b7ba-e818605e5c76.png)<br/>


# **css color link** <br/>

[**w3schools**](https://www.w3schools.com/cssref/css_colors.asp): <br/>


# **plot marker** <br/>
```
BTC['signal'] = np.where(
    BTC['close'] > BTC['sam15'], 1, 0)

BTC['position'] = BTC['signal'].diff()

BTC['long'] = np.where(
    (BTC['position']) == 1, BTC.close, np.NAN)
BTC['short'] = np.where(
    (BTC['position']) == -1, BTC.close, np.NAN)

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['long'],
                         mode='markers',
                         marker=dict(color='DarkGreen', size=12,
                                     opacity=1),
                         marker_symbol=5,
                         name='long signal'))

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['short'],
                         mode='markers',
                         marker=dict(color='MediumVioletRed', size=12,
                                     opacity=1),
                         marker_symbol=6,
                         name='short signal'))
```
![newplot (8)](https://user-images.githubusercontent.com/102425717/179375646-9d1bdf05-f476-4ff5-917a-3dd3ceaf74ac.png)<br/>

# **marker Styl link** <br/>

[**plotly**](https://plotly.com/python/marker-style/): <br/>


# **plot volume in subplot** <br/>

```
# first replace    fig = go.Figure() replace 
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

#add
colors = ['green' if row['open'] - row['close'] >= 0
          else 'red' for index, row in BTC.iterrows()]
fig.add_trace(go.Bar(x=BTC.index,
                     y=BTC['volume'],
                     marker_color=colors
                     ), row=2, col=1)
```
![newplot (9)](https://user-images.githubusercontent.com/102425717/179375789-94caf75c-f00f-4934-bc11-b8b53cab885c.png)<br/>

# **visible plot** <br/>

```
# add visible='legendonly'

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['short'],
                         mode='markers',
                         marker=dict(color='MediumVioletRed', size=12,
                                     opacity=1),
                         marker_symbol=6,
                         name='short signal',
                         visible='legendonly'))
```

# **fill between two plot lilke BBband** <br/>

```
# { bollinger band
def SMA(data: str = 'BTC', src: str = 'close', length: int = 20):  # sma for middle band
    return data[src].rolling(window=length).mean()


def TP(data: str = 'BTC'):  # hlc/3 (typical price)
    return(data['high']+data['low']+data['close'])/3


def BOLU(data: str = "BTC", src: str = 'tp', length: int = 20, m: int = 2):  # upperband
    return SMA(data, src, 20)+((m)*data[src].rolling(window=length).std())


def BOLD(data: str = 'BTC', src: str = 'tp', length: int = 20, m: int = 2):  # lower band
    return SMA(data, 'close', 20)-((m)*data[src].rolling(window=length).std())
# ​}

# { use bollinger band functions to calculate  bbband
BTC['middelband'] = SMA(BTC, 'close', 20)
BTC['tp'] = TP(BTC)
BTC['upperband'] = BOLU(BTC, 'tp')
BTC['lowerband'] = BOLD(BTC, 'tp')
# }

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['upperband'],
                         fill=None,
                         mode='lines',
                         line_color='rgba(0, 0, 255, 0.5)',
                         line=dict(width=2),
                         name='upperband',
                         visible='legendonly'))

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['lowerband'],
                         opacity=0.3,
                         fill='tonexty',
                         line=dict(width=2),
                         name='lowerband',
                         line_color='rgba(0, 0, 255, 0.5)',
                         mode='lines', fillcolor='rgba(0, 0, 255, 0.1)',
                         visible='legendonly'))
```
![newplot (13)](https://user-images.githubusercontent.com/102425717/179430133-86791680-9c85-4d57-9b8f-984a6713e3bf.png) <br/>

# **css rgba color link** <br/>

[**w3schools**](https://www.w3schools.com/css/css_colors_rgb.asp): <br/>

# **change candlestack color with condition** <br/>
```
# { set condition in new dataframe 
bulish = BTC[(BTC['in_sqz'] == False) & (
    BTC['close'] > BTC['middelband'])]
not_bulish = BTC[BTC.index.isin(bulish.index)].index
# }

fig.add_traces(go.Candlestick(x=billish.index,
                              open=bulish['open'], high=bulish['high'],
                              low=bulish['low'], close=bulish['close'],
                              increasing_line_color='SpringGreen',
                              decreasing_line_color='DarkGreen',
                              name='Bulish momentum(+/-)'))
```
![newplot (14)](https://user-images.githubusercontent.com/102425717/179430389-ada8f90b-93a8-411e-9234-f04f51349936.png) <br/>


# **finaly code** <br/>
```
# { import the libraries
import ccxt
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
# }

# { show all rows and column
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_column', None)
# }

# { load exchange
exchange = ccxt.binance({
    'options': {
        'adjustForTimeDifference': True,
    },

})
# }


# { load data as function
def fetch(symbol: str, timeframe: str, limit: int):
    print(f"Fetching {symbol} new bars for {datetime.now().isoformat()}")

    bars = exchange.fetch_ohlcv(
        symbol, timeframe=timeframe, limit=limit)  # fetch ohlcv
    df = pd.DataFrame(bars[:-1], columns=['timestamp',
                      'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index(pd.DatetimeIndex(df.timestamp))
    return df
# }


# { bollinger band
def SMA(data: str = 'BTC', src: str = 'close', length: int = 20):  # sma for middle band
    return data[src].rolling(window=length).mean()


def TP(data: str = 'BTC'):  # hlc/3 (typical price)
    return(data['high']+data['low']+data['close'])/3


def BOLU(data: str = "BTC", src: str = 'tp', length: int = 20, m: int = 2):  # upperband
    return SMA(data, src, 20)+((m)*data[src].rolling(window=length).std())


def BOLD(data: str = 'BTC', src: str = 'tp', length: int = 20, m: int = 2):  # lower band
    return SMA(data, 'close', 20)-((m)*data[src].rolling(window=length).std())
# ​}


# { set the symbol for data function
BTC = fetch('BTC/USDT', '1h', 1000)
# }


# { use panda ta lib for calculate sma
BTC['sam15'] = ta.sma(BTC.close, 15)
# }


# { creat tow data frame for change sma color
BTC.loc[BTC['sam15'] >= BTC['close'], 'upper_sma15'] = BTC['sam15']
BTC.loc[BTC['sam15'] < BTC['close'], 'lower_sma15'] = BTC['sam15']
# }


# { use bollinger band functions to calculate  bbband
BTC['middelband'] = SMA(BTC, 'close', 20)
BTC['tp'] = TP(BTC)
BTC['upperband'] = BOLU(BTC, 'tp')
BTC['lowerband'] = BOLD(BTC, 'tp')
# }


# { find cross for plot marker
BTC['signal'] = np.where(
    BTC['close'] > BTC['sam15'], 1, 0)

BTC['position'] = BTC['signal'].diff()

BTC['long'] = np.where(
    (BTC['position']) == 1, BTC.close, np.NAN)
BTC['short'] = np.where(
    (BTC['position']) == -1, BTC.close, np.NAN)
# }


# { set condition in new dataframe
bulish = BTC[(BTC['close'] > BTC['middelband'])]
not_bulish = BTC[BTC.index.isin(bulish.index)].index
# }


# {  plot the data
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
#fig = go.Figure()

fig.add_trace(go.Candlestick(x=BTC.index,
                             open=BTC['open'],
                             high=BTC['high'],
                             low=BTC['low'],
                             close=BTC['close'],
                             showlegend=False))

fig.add_traces(go.Candlestick(x=bulish.index,
                              open=bulish['open'], high=bulish['high'],
                              low=bulish['low'], close=bulish['close'],
                              increasing_line_color='SpringGreen',
                              decreasing_line_color='DarkGreen',
                              name='Bulish momentum(+/-)'))

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['upper_sma15'],
                         opacity=1,
                         line=dict(color='Lime', width=4),
                         name='upper_sma'))

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['lower_sma15'],
                         opacity=1,
                         line=dict(color='Maroon', width=4),
                         name='lower_sma'))


fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['long'],
                         mode='markers',
                         marker=dict(color='DarkGreen', size=12,
                                     opacity=1),
                         marker_symbol=5,
                         name='long signal'))

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['short'],
                         mode='markers',
                         marker=dict(color='MediumVioletRed', size=12,
                                     opacity=1),
                         marker_symbol=6,
                         name='short signal',
                         visible='legendonly'))

colors = ['green' if row['open'] - row['close'] >= 0
          else 'red' for index, row in BTC.iterrows()]
fig.add_trace(go.Bar(x=BTC.index,
                     y=BTC['volume'],
                     marker_color=colors
                     ), row=2, col=1)

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['middelband'],
                         opacity=1,
                         line=dict(color='orange', width=2),
                         name='middelband'))

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['upperband'],
                         fill=None,
                         mode='lines',
                         line_color='rgba(0, 0, 255, 0.5)',
                         line=dict(width=2),
                         name='upperband'))

fig.add_trace(go.Scatter(x=BTC.index,
                         y=BTC['lowerband'],
                         opacity=0.3,
                         fill='tonexty',
                         line=dict(width=2),
                         name='lowerband',
                         line_color='rgba(0, 0, 255, 0.5)',
                         mode='lines', fillcolor='rgba(0, 0, 255, 0.1)'))

fig.show()
# }

```

![newplot (15)](https://user-images.githubusercontent.com/102425717/179430447-eaa8a175-f08c-4cf4-a7f3-dd64a7ca4ed3.png) <br/>

