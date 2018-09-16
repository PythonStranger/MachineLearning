import numpy as np
import pandas as pd

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import datetime

import matplotlib.pyplot as plt

from sklearn import linear_model as lm, metrics, model_selection

import talib as ta

# Get data
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2018, 8, 1)
df = web.DataReader('BABA', 'iex', start, end)
df = df.dropna()
df = df.iloc[:, :4]

# Calculate some technical indicator
# df['MA_10'] = df['close'].rolling(window=10).mean()
# df['Corr'] = df['close'].rolling(window=10).corr(df['MA_10'])
# df['open-close'] = df['open'] - df['close'].shift(1)
df['ADX'] = ta.ADX(np.array(df['high']), np.array(df['low']),
                   np.array(df['close']), timeperiod=14)
df['BOP'] = ta.BOP(np.array(df['open']),np.array(df['high']),np.array(df['low']),
                   np.array(df['close']))
df['CCI'] = ta.CCI(np.array(df['high']), np.array(df['low']),
                   np.array(df['close']), timeperiod=14)

# List X and y
df = df.dropna()
X = df.iloc[:, :7]
y = np.where((df['close'].shift(-1) > df['close']), 1, -1)

# Run the Logistic Regression
# Taking training set and test set to 7/3

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y.ravel(),
                                                                    random_state=7, shuffle=False)

# LogisticRegression
model = lm.LogisticRegression()
model = model.fit(X_train, y_train)

# Run model in the test set

probability = model.predict_proba(X_test)
predicted = model.predict(X_test)

# Performance reports
print('Confusion matrix:')
print(metrics.confusion_matrix(y_test, predicted))
print('Classification reports:')
print(metrics.classification_report(y_test, predicted))
print('Model score:')
print(model.score(X_test, y_test))

# Do cross validation
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=False)
cv_results = model_selection.cross_val_score(model, X, y.ravel(),
                                             cv=kfold, scoring='accuracy')
print('cv mean:')
print(cv_results.mean())
print('cv std:')
print(cv_results.std())

df['Signal'] = model.predict(X)

# log returns on the stock
df['Returns'] = np.log(df['close'] / df['close'].shift(1))
split = int(0.7 * len(df))
Cum_Returns = np.cumsum(df[split:]['Returns'])

# Using strategy which is long if signal = 1, otherwise short
df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift(1)
Cum_Strategy_Returns = np.cumsum(df[split:]['Strategy_Returns'])

# Plot
plt.figure()
plt.plot(Cum_Returns, color='r', label='Original Returns')
plt.plot(Cum_Strategy_Returns, color='g', label='Strategy Returns')
# plt.xlabel(df.index[split:])
plt.legend()
plt.show()
