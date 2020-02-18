import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
import pandas as pd
from linearRegression.linearRegression import LinearRegression
import copy

mag = []
N = [i for i in range(60,300*j,4)]
degree = []
for j in range(1,7):
    the = []
    for i in range(1,10,2):
        x = np.array([i*np.pi/180 for i in range(60,300*j,4)])
        np.random.seed(10)  
        y = 4*x + 7 + np.random.normal(0,3,len(x))
        y = pd.Series(y)
        poly = PolynomialFeatures(degree=i)
        X = poly.transform(x)
        LR = copy.deepcopy(LinearRegression(fit_intercept=False))
        LR.fit_non_vectorised(X, y, n_iter=5 ,batch_size=60*j)
        coef = LR.coef_
        the.append(np.linalg.norm(coef))
    mag.append(the)
mag = np.array(mag)
N = np.array(N)
degree = np.array([i for i in range(1,10,2)])
N = [(300*i-60)//4 for i in range(1,7)]
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(N)):
    ax.set_yscale('log')
    ax.plot(degree, mag[i], label='N='+str(N[i]))
ax.set_xlabel('degree')
ax.set_ylabel('Magnitude of theta')
ax.set_title('Magnitude of theta (log scale) for different N and degree')
ax.legend()
plt.show()

