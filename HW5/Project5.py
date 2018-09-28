import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
import talib as ta

df = pd.read_csv('~/Downloads/QDA_Strategy_Data.csv')
# Calculate the RSI and RSI MA
df['RSI'] = ta.RSI(np.array(df['Close']), timeperiod=10)
df['RSI_MA'] = df['RSI'].rolling(10).mean()

# Get ADX which I'd like to use as features later
df['ADX'] = ta.ADX(np.array(df['High']), np.array(df['Low']),
                   np.array(df['Close']), timeperiod=14)

# Create crossover signal column out of the indicators
df = df.assign(Signal=pd.Series(np.zeros(len(df))).values)
df.loc[df['RSI'] < df['RSI_MA'], 'Signal'] = 1
df.loc[df['RSI'] > df['RSI_MA'], 'Signal'] = -1

df['Return'] = np.log(df['Close'] / df['Close'].shift(1))  # Calc log return
df['S_Return'] = df['Signal'].shift(1) * df['Return']  # Signal times the return
df['S_Vol'] = df['S_Return'].rolling(5).std()
df['Market_Return'] = df['Return'].expanding().sum()

df['Return Direction'] = np.where(df['S_Return'] > df['S_Vol'], 'UP',
                                  np.where(df['S_Return'] < -df['S_Vol'], 'DOWN', 'FLAT'))
# Add a volatility and lagged volatility features
df['Vol'] = df['Close'].rolling(window=5).std()
df['Vol Lag 3'] = df['Vol'].shift(3)
df['Vol Lag 4'] = df['Vol'].shift(4)
df['Vol Lag 5'] = df['Vol'].shift(5)
# Use these volatility features and RSI as the feature set
X = df[['Vol Lag 3', 'Vol Lag 4', 'ADX', 'Return']]
y = df['Return Direction']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Use QDA to predict UP, DOWN, or FLAT
model = QuadraticDiscriminantAnalysis()
model.fit(X_train.fillna(0), y_train)
predictions = model.predict(X_test.fillna(0))
# Check the confusion matrix and classification report
print('\nQDA CONFUSION MATRIX:\n')
print(confusion_matrix(y_test, predictions))
print('\nQDA CLASSIFICATION REPORT:\n')
print(classification_report(y_test, predictions))
# Initializing second predictions variable for the data
df['Predictions'] = model.predict(X.fillna(0))
# print( df[ 'Predictions' ] )
# Change the signal so that DOWN predictions are 0, instead of -1
df['QDA Signal'] = np.zeros(len(df))
df['QDA Signal'] = np.where(df['Predictions'] == 'DOWN', 0, df['Signal'])
# This time use the QDA signal instead to backtest the strategy
df['SQ_Return'] = df['QDA Signal'].shift(1) * df['Return']
df['Strategy_Return'] = df['SQ_Return'].expanding().sum()
# Create trading performance metrics
df['Wins'] = np.where(df['SQ_Return'] > 0, 1, 0)
df['Losses'] = np.where(df['SQ_Return'] < 0, 1, 0)
df['Total Wins'] = df['Wins'].sum()
df['Total Losses'] = df['Losses'].sum()
df['Total Trades'] = df['Total Wins'][0] + df['Total Losses'][0]
df['Hit Ratio'] = round(df['Total Wins'] / df['Total Losses'], 2)
df['Win Pct'] = round(df['Total Wins'] / df['Total Trades'], 2)
df['Loss Pct'] = round(df['Total Losses'] / df['Total Trades'], 2)
# Plot the performance of the RSI Strategy
plt.plot(df['Market_Return'], color='black', label='Market Returns')
plt.plot(df['Strategy_Return'], color='blue', label='Strategy Returns')
plt.legend(loc=0)
plt.tight_layout()
plt.show()
print('Hit Ratio:', df['Hit Ratio'][0])
print('Win Percentage:', df['Win Pct'][0])
print('Loss Percentage:', df['Loss Pct'][0])
print('Total Trades:', df['Total Trades'][0])
