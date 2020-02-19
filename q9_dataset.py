import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)
import copy
N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

# print(X)
print("===============================================")
print("For fit_vectorised")
print("===============================================")
# for lr_type in ["constant", "inverse"]:
#     for fit_intercept in [True, False]:
print('Random data')
fit_intercept = True
LR = copy.deepcopy(LinearRegression(fit_intercept=fit_intercept))
LR.fit_vectorised(X, y, batch_size=30) # here you can use fit_non_vectorised / fit_autograd methods
y_hat = LR.predict(X)
# print(LR.thetas)
print("--------------------------------------------------")
print("fit_intercept : "+str(fit_intercept))
print("--------------------------------------------------")
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print("--------------------------------------------------")
X[5] = X[0]+5
X[6] = X[1]*3
print('Data with multicollinearity')
fit_intercept = True
LR = copy.deepcopy(LinearRegression(fit_intercept=fit_intercept))
LR.fit_vectorised(X, y, batch_size=30) # here you can use fit_non_vectorised / fit_autograd methods
y_hat = LR.predict(X)
# print(LR.thetas)
print("--------------------------------------------------")
print("fit_intercept : "+str(fit_intercept))
print("--------------------------------------------------")
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))