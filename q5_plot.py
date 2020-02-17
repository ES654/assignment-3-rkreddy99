import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
import pandas as pd
from linearRegression.linearRegression import LinearRegression
from metrics import *

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  
y = 4*x + 7 + np.random.normal(0,3,len(x))
y = pd.Series(y)
# print("===============================================")
# print("For fit_vectorised")
# print("===============================================")
# # for lr_type in ["constant", "inverse"]:
for i in range(1,2):
    poly = PolynomialFeatures(degree=i)
    X = poly.transform(x.copy())
    # print(X[1].loc[23])
    LR = LinearRegression(fit_intercept=False)
    LR.fit_non_vectorised(X, y, batch_size=6) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    LR.plot_surface(np.array(X[1]),y)
    LR.plot_line_fit(np.array(X[1]), np.array(y), 3,4)
    print("--------------------------------------------------")
    # print("fit_intercept : "+str(fit_intercept))
    # print("degree : "+str(i))
    # print("--------------------------------------------------")
    # print('RMSE: ', rmse(y_hat, y))
    # print('MAE: ', mae(y_hat, y))