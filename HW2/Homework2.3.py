import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.manifold import TSNE

# Get Data
td = pd.read_csv('/Users/zhengtianxiang/Desktop/courses/Computer science/data/Futures Data.csv', index_col='Date')

# Deal with data
data = (np.log(td) - np.log(td.shift(1))).dropna()
X = data.drop(data.index[-1])
y = data['SPX'][1:]

# t_SNE to get components
DRmodel = TSNE(perplexity=30)
components3 = DRmodel.fit_transform(X)

# Do linear regression and get R^2, MSE, MAE
model3 = LinearRegression()
model3.fit(components3, y)
y_pred3 = model3.predict(components3)
print('\nR-Squared: ', metrics.explained_variance_score(y, y_pred3))
print('Polynomial MSE:', metrics.mean_squared_error(y, y_pred3))
print('Polynomial MAE:', metrics.mean_absolute_error(y, y_pred3), )
print('\n')
