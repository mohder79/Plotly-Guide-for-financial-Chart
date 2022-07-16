# Plotly-Guide-for-Financial-Chart




# import libraries  <br/>

```
import ccxt
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

```



# **load exchange** <br/>

```

exchange = ccxt.binance({
    'options': {
        'adjustForTimeDifference': True,
    },

})

```

