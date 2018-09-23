import numpy as np
import pandas as pd

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
import talib as ta


# Get data
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2018, 8, 1)
df = web.DataReader('BABA', 'iex', start, end)
df = df.dropna()
df = df.iloc[:, :4]

df['ADX'] = ta.ADX(np.array(df['high']), np.array(df['low']),
                   np.array(df['close']), timeperiod=14)
df['BOP'] = ta.BOP(np.array(df['open']),np.array(df['high']),np.array(df['low']),
                   np.array(df['close']))
df['CCI'] = ta.CCI(np.array(df['high']), np.array(df['low']),
                   np.array(df['close']), timeperiod=14)


df = df.assign( Signal = pd.Series( np.zeros( len( df ) ) ).values )
df[ 'Vol' ] = df[ 'close' ].rolling(5).std()
df['Move'] =  df[ 'close' ] - df[ 'close' ].shift( 1 )
df.loc[df['close'] > df['close'].shift(1), 'Signal'] = 1  # Long signal
df.loc[df['close'] < df['close'].shift(1), 'Signal'] = -1  # Short signal
df[ 'Return Direction' ] = np.where( df[ 'Move' ] > df['Vol'], 'UP', np.where(df[ 'Move' ] < -df['Vol'],'DOWN', 'FLAT' ))


#df.loc[ df[ 'ADX' ] < df[ 'ADX_MA' ], 'Signal' ] = 1 # Long signal
#df.loc[ df[ 'ADX' ] > df[ 'ADX_MA' ], 'Signal' ] = -1 # Short signal
# Backtest the signal
df[ 'Return' ] = np.log( df[ 'close' ] / df[ 'close' ].shift( 1 )) # Calc log return
#df['Move'] =  df[ 'Close' ] - df[ 'Close' ].shift( 1 )
#df[ 'S_Return' ] = df[ 'Signal' ]* df[ 'Return' ] # Signal times the return
#df['S_Move'] =  df[ 'Signal' ]* df[ 'Move' ]
df[ 'Market_Return' ] = df[ 'Return' ].expanding().sum()
# Classify strategy returns as UP, DOWN, or FLAT


#df[ 'Return Direction' ] = np.where( df[ 'S_Move' ] > df['Vol'], 'UP', np.where(df[ 'S_Move' ] < -df['Vol'],'DOWN', 'FLAT' ))
# Add a volatility and lagged volatility features

#df[ 'Vol Lag 3' ] = df[ 'Vol' ].shift( 3 )
#df['Vol Lag 4'] = df['Vol'].shift(4)
#df['Vol Lag 5'] = df['Vol'].shift(5)
# Use these volatility features and RSI as the feature set
X = df[ [ 'Vol', 'ADX', 'BOP', 'CCI' ] ]
y = df[ 'Return Direction' ]
# Split the data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2 )

model = QuadraticDiscriminantAnalysis()
model.fit( X_train.fillna( 0 ), y_train )
predictions = model.predict( X_test.fillna( 0 ) )

# Check the confusion matrix and classification report
print( '\nQDA CONFUSION MATRIX:\n' )
print( confusion_matrix( y_test, predictions ) )
print( '\nQDA CLASSIFICATION REPORT:\n' )
print( classification_report( y_test, predictions ) )

df[ 'Predictions' ] = model.predict( X.fillna( 0 ) )
df[ 'QDA Signal' ] = np.zeros( len( df ) )
df[ 'QDA Signal' ] = np.where( df[ 'Predictions' ] == 'DOWN', 0, df[ 'Signal' ] )

df[ 'SQ_Return' ] = df[ 'QDA Signal' ] * df[ 'Return' ]
df[ 'Strategy_Return' ] = df[ 'SQ_Return' ].expanding().sum()

df[ 'Wins' ] = np.where( df[ 'SQ_Return' ] > 0, 1, 0 )
df[ 'Losses' ] = np.where( df[ 'SQ_Return' ] < 0, 1, 0 )
df[ 'Total Wins' ] = df[ 'Wins' ].sum()
df[ 'Total Losses' ] = df[ 'Losses' ].sum()
df[ 'Total Trades' ] = df[ 'Total Wins' ][ 0 ] + df[ 'Total Losses' ][ 0 ]
df[ 'Hit Ratio' ] = round( df[ 'Total Wins' ] / df[ 'Total Losses' ], 2 )
df[ 'Win Pct' ] = round( df[ 'Total Wins' ] / df[ 'Total Trades' ], 2 )
df[ 'Loss Pct' ] = round( df[ 'Total Losses' ] / df[ 'Total Trades' ], 2 )


plt.plot( df[ 'Market_Return' ], color = 'black', label = 'Market Returns' )
plt.plot( df[ 'Strategy_Return' ], color = 'blue', label = 'Strategy Returns' )
plt.legend( loc = 0 )
plt.tight_layout()
plt.show()
print( 'Hit Ratio:', df[ 'Hit Ratio' ][ 0 ] )
print( 'Win Percentage:', df[ 'Win Pct' ][ 0 ] )
print( 'Loss Percentage:', df[ 'Loss Pct' ][ 0 ] )
print( 'Total Trades:', df[ 'Total Trades' ][ 0 ] )