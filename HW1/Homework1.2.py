import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from sklearn import metrics
import warnings


# LinearRegression
def Linear_Regression(X_temp, y_Return, y_Return_N):
    lm = LinearRegression()
    lm.fit(X_temp, y_Return)
    prediction = lm.predict(X_temp)
    lm_N = LinearRegression()
    lm_N.fit(X_temp, y_Return_N)
    prediction_N = lm.predict(X_temp)

    print("for Annual Returns's coeffience are \n", [lm.intercept_] + lm.coef_)
    print("for Annual Returns_N's coeffience are \n", [lm_N.intercept_] + lm_N.coef_)
    # error
    print('error for Annual return and Annual return_N')
    print('Polynomial MSE:', metrics.mean_squared_error(y_Return, prediction),
          metrics.mean_squared_error(y_Return_N, prediction_N))
    print('Polynomial MAE:', metrics.mean_absolute_error(y_Return, prediction),
          metrics.mean_absolute_error(y_Return_N, prediction_N))
    print('Polynomial RMSE:', np.sqrt(metrics.mean_squared_error(y_Return, prediction)),
          np.sqrt(metrics.mean_squared_error(y_Return_N, prediction_N)))
    print('\n')
    return


# LassoRegression
def Lasso_Regression(X_temp, y_Return, y_Return_N):
    import pandas as pd
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    import numpy as np
    from sklearn import metrics
    import warnings
    a = .005
    maxiters = 50000
    model = Lasso(alpha=a, max_iter=maxiters)
    model.fit(X_temp, y_Return)
    prediction = model.predict(X_temp)
    model_N = Lasso(alpha=a, max_iter=maxiters)
    model_N.fit(X_temp, y_Return_N)
    prediction_N = model_N.predict(X_temp)

    print("for Annual Returns's coeffience are \n", [model.intercept_] + model.coef_)
    print("for Annual Returns_N's coeffience are \n", [model_N.intercept_] + model_N.coef_)
    # error
    print('error for Annual return and Annual return_N')
    print('Polynomial MSE:', metrics.mean_squared_error(y_Return, prediction),
          metrics.mean_squared_error(y_Return_N, prediction_N))
    print('Polynomial MAE:', metrics.mean_absolute_error(y_Return, prediction),
          metrics.mean_absolute_error(y_Return_N, prediction_N))
    print('Polynomial RMSE:', np.sqrt(metrics.mean_squared_error(y_Return, prediction)),
          np.sqrt(metrics.mean_squared_error(y_Return_N, prediction_N)))
    print('\n')
    return


# RidgeRegression
def Ridge_Regression(X_temp, y_Return, y_Return_N):
    import pandas as pd
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    import numpy as np
    from sklearn import metrics
    import warnings
    a = .005
    model = Ridge(alpha=a)
    model.fit(X_temp, y_Return)
    prediction = model.predict(X_temp)
    model_N = Ridge(alpha=a)
    model_N.fit(X_temp, y_Return_N)
    prediction_N = model_N.predict(X_temp)

    print("for Annual Returns's coeffience are \n", [model.intercept_] + model.coef_)
    print("for Annual Returns_N's coeffience are \n", [model_N.intercept_] + model_N.coef_)
    # error
    print('error for Annual return and Annual return_N')
    print('Polynomial MSE:', metrics.mean_squared_error(y_Return, prediction),
          metrics.mean_squared_error(y_Return_N, prediction_N))
    print('Polynomial MAE:', metrics.mean_absolute_error(y_Return, prediction),
          metrics.mean_absolute_error(y_Return_N, prediction_N))
    print('Polynomial RMSE:', np.sqrt(metrics.mean_squared_error(y_Return, prediction)),
          np.sqrt(metrics.mean_squared_error(y_Return_N, prediction_N)))
    print('\n')
    return


if __name__ == '__main__':
    # Import data
    data = pd.read_csv(r'/Users/zhengtianxiang/Desktop/courses/Computer science/data/Stock Performance.csv',
                       parse_dates=True)
    # Dealing with data
    X = data.loc[:, ('Large BP', 'Large ROE', 'Large SP', 'Large Return Rate', 'Small Systematic Risk')]
    X.loc[:, 'constant'] = 1
    X_temp = X.loc[:, ('constant', 'Large BP', 'Large ROE', 'Large SP', 'Large Return Rate', 'Small Systematic Risk')]
    y_Return = data['Annual Return']
    y_Return_N = data['Annual Return N']

    # Regression
    print('Linear regression using sklearn.linear_model.LinearRegression')
    Linear_Regression(X_temp, y_Return, y_Return_N)
    print('\nLinear regression using sklearn.linear_model.Lasso')
    Lasso_Regression(X_temp, y_Return, y_Return_N)
    print('\nLinear regression using sklearn.linear_model.Ridge')
    Ridge_Regression(X_temp, y_Return, y_Return_N)

    print('From the result, The Ridge_Regression performs best!')
