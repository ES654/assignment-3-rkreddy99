import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
import pandas as pd
from linearRegression.linearRegression import LinearRegression
import math

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  
y = 4*x + 7 + np.random.normal(0,3,len(x))
y = pd.Series(y)
mag = []
degree = []
for i in range(1,11):
    poly = PolynomialFeatures(degree=i)
    X = poly.transform(x.copy())
    LR = LinearRegression(fit_intercept=False)
    LR.fit_non_vectorised(X, y, n_iter=5 ,batch_size=60)
    coef = LR.coef_
    mag.append(np.linalg.norm(coef))
    degree.append(i)
mag = np.array(mag)
degree = np.array(degree)
plt.plot(degree, mag)
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Magnitude of theta')
plt.title('Magnitude of theta (log scale) vs Degree')
plt.show()

