import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import warnings
warnings.filterwarnings("ignore")
#get data
DATAPATH = 'weatherAUS_ch.csv'
data = pd.read_csv(DATAPATH)
data.head()

#linear regression
x5 = data.drop(['RainTomorrow'], axis=1)
y = data['RainTomorrow'].values.reshape(-1,1)
lin_reg = LinearRegression()
MSE5 = cross_val_score(lin_reg,x5,y,scoring='neg_mean_squared_error', cv=5)
mean_MSE = np.mean(MSE5)
print(mean_MSE)

#ridge
ridge = Ridge()
parameters = {'alpha': [1e-100, 1e-50, 1e-30, 1e-15, 1e-5, 1e-1, 1, 5, 10, 20, 50, 100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(x5,y)
print("ridge regressor best lamda :", ridge_regressor.best_params_)
print("ridge regressor best MSE : ", ridge_regressor.best_score_)

#lasso
lasso = Lasso()
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(x5,y)
print("lasso regressor best lamda :",lasso_regressor.best_params_)
print("lasso regressor best MSE : ", lasso_regressor.best_score_)

#elastic net
elastic =ElasticNet(normalize=True)
elastic_regressor=GridSearchCV(estimator=elastic,param_grid=parameters,scoring='neg_mean_squared_error', cv=5)
elastic_regressor.fit(x5,y)
print("elastic net regressor best lamda :",elastic_regressor.best_params_)
print("elastic net regressor best MSE : ", elastic_regressor.best_score_)