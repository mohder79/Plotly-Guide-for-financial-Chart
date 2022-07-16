# Plotly-Guide-for-Financial-Chart




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

# **OHLC Chart** <br/>
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


# **removing rangeslider** <br/>
```
fig.update_layout(xaxis_rangeslider_visible=False)
```
# **hide weekends** <br/>
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

# [**ema**](https://github.com/mohder79/indicators/blob/main/ema.py): <br/>

# **css color link** <br/>

[**w3schools**](https://www.w3schools.com/cssref/css_colors.asp): <br/>
