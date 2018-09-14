import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Get Data
td = pd.read_csv('/Users/zhengtianxiang/Desktop/courses/Computer science/data/Futures Data.csv', index_col='Date')

# Deal with data
data = (np.log(td) - np.log(td.shift(1))).dropna()
X = data.drop(data.index[-1])
y = data['SPX'][1:]

# PCA to get components
DRmodel = PCA(n_components=9)
components1 = DRmodel.fit_transform(X)

# Do linear regression and get R^2, MSE, MAE
dimensional_two = components1[:, :2]
model1 = LinearRegression()
model1.fit(dimensional_two, y)
y_pred1 = model1.predict(dimensional_two)
print('\nR-Squared: ', metrics.explained_variance_score(y, y_pred1))
print('Polynomial MSE:', metrics.mean_squared_error(y, y_pred1))
print('Polynomial MAE:', metrics.mean_absolute_error(y, y_pred1), )
print('\n')

print("success connect to the github!")