import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


print("===============================================")
print("For fit_vectorised")
print("===============================================")
for lr_type in ["constant", "inverse"]:
    for fit_intercept in [True, False]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_vectorised(X, y, batch_size=len(X.index), lr_type=lr_type) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)
        print("--------------------------------------------------")
        print("fit_intercept : "+str(fit_intercept)+" && lr_type : "+str(lr_type))
        print("--------------------------------------------------")
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))
print("===============================================")
print("For fit_non_vectorised")
print("===============================================")
for lr_type in ["constant", "inverse"]:
    for fit_intercept in [True, False]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_non_vectorised(X, y, batch_size=len(X.index),lr_type=lr_type) # here you can use fit_non_vectorised / fit_autograd methods
        y_hat = LR.predict(X)
        print("--------------------------------------------------")
        print("fit_intercept : "+str(fit_intercept)+" && lr_type : "+str(lr_type))
        print("--------------------------------------------------")
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))

