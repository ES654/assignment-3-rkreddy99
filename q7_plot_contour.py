import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from preprocessing.polynomial_features import PolynomialFeatures
import copy

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  
y = 4*x + 7 + np.random.normal(0,3,len(x))
y = pd.Series(y)
poly = PolynomialFeatures(degree=1)
X = poly.transform(x.copy())

LR1 = copy.deepcopy(LinearRegression(fit_intercept=False))
LR1.fit_non_vectorised(X, y, batch_size=6)
LR1.fit = 'non_vectorised'
y_hat = LR1.predict(X)
LR1.plot_surface(np.array(X[1]),y)
LR1.plot_line_fit(np.array(X[1]), np.array(y),t_0 = 1, t_1 = 1)
LR1.plot_contour(np.array(X[1]), np.array(y),t_0 = 1, t_1 = 1)
print("--------------------------------------------------")

LR2 = copy.deepcopy(LinearRegression(fit_intercept=False))
LR2.fit_vectorised(X, y, batch_size=60)
LR2.fit = 'vectorised'
y_hat = LR2.predict(X)
LR2.plot_surface(np.array(X[1]),y)
LR2.plot_line_fit(np.array(X[1]), np.array(y),t_0 = 1, t_1 = 1)
LR2.plot_contour(np.array(X[1]), np.array(y),t_0 = 1, t_1 = 1)
print("--------------------------------------------------")

LR3 = copy.deepcopy(LinearRegression(fit_intercept=False))
LR3.fit_autograd(X, y, batch_size=6)
LR3.fit = 'autograd'
y_hat = LR3.predict(X)
LR3.plot_surface(np.array(X[1]),y)
LR3.plot_line_fit(np.array(X[1]), np.array(y),t_0 = 1, t_1 = 1)
LR3.plot_contour(np.array(X[1]), np.array(y),t_0 = 1, t_1 = 1)
print("--------------------------------------------------")    
