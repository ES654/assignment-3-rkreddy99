import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import copy
from metrics import *
import time

np.random.seed(42)
N = [1000*i for i in range(1,5)]
P = 5
grad_time = []
norm_time = []
for i in range(len(N)):
  X = pd.DataFrame(np.random.randn(N[i], P))
  y = pd.Series(np.random.randn(N[i]))
  LR = copy.deepcopy(LinearRegression(fit_intercept=True))
  a = time.time()
  LR.fit_vectorised(X.copy(),y.copy(),n_iter = 5,batch_size=N[i])
  LR.predict(X.copy())
  b = time.time()
  LR = copy.deepcopy(LinearRegression(fit_intercept=True))
  c = time.time()
  LR.fit_normal(X.copy(),y.copy())
  LR.predict(X.copy())
  d = time.time()
  grad_time.append(b-a)
  norm_time.append(d-c)
#   print(grad_time, norm_time)
grad_time, norm_time = np.array(grad_time), np.array(norm_time)
N = np.array(N)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(N,grad_time, label = 'gradient descent')
ax.plot(N, norm_time, label = 'normal')
ax.set_ylabel('time(s)')
ax.set_xlabel('No. of samples')
ax.set_title('Gradient Descent vs Normal, batch_size=N, n_iter=5, P=5')
ax.legend()
plt.show()
