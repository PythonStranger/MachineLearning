from sklearn.linear_model import LinearRegression
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

# Got MCD and S&P500 data from 1/1/2016-7/31/2018
MCD = pd.read_csv(r'/Users/zhengtianxiang/Desktop/courses/Computer science/data/MCD.csv', parse_dates=True,
                  index_col='Date', )
SP_500 = pd.read_csv(r'/Users/zhengtianxiang/Desktop/courses/Computer science/data/^GSPC.csv', parse_dates=True,
                     index_col='Date')

# Joint the closing price of the two datasets
column = ['MCD', 'S', '&P500']
close_price = pd.concat([MCD.loc[:,'Close'], SP_500.loc[:,'Close']], axis=1)
close_price.columns = column

# Calculate returns
returns = close_price.pct_change()
clean_returns = returns.dropna()
# clean_returns.loc[:,'constant'] = 1
constant = pd.Series(np.ones(np.shape(clean_returns)[0]), name='constant', index=clean_returns.index)
data = pd.concat([clean_returns, constant], axis=1)

# Now, Start to regression
# Taking S&P500 as X, MCD as y
X = data[['constant', 'S&P500']]
y = data['MCD']

# linear regression using scikitlearn
lm = LinearRegression()
lm.fit(X, y)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

Beta = lm.coef_[1]
print(Beta)
