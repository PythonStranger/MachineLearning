import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Get Data
td = pd.read_csv('/Users/zhengtianxiang/Desktop/courses/Computer science/data/Futures Data.csv', index_col='Date')

# Deal with data
data = (np.log(td) - np.log(td.shift(1))).dropna()
X = data.drop(data.index[-1])
y = data['SPX'][1:]

# SVD to get components
svd = TruncatedSVD(n_components=2)
svd.fit(X)
components2 = svd.transform(X)

# Do linear regression and get R^2, MSE, MAE
model2 = LinearRegression()
model2.fit(components2, y)
y_pred2 = model2.predict(components2)
print('\nR-Squared: ', metrics.explained_variance_score(y, y_pred2))
print('Polynomial MSE:', metrics.mean_squared_error(y, y_pred2))
print('Polynomial MAE:', metrics.mean_absolute_error(y, y_pred2), )
print('\n')
