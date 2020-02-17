import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
import autograd.numpy as np1
from autograd import grad
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import time
class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

    
    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        assert (batch_size <= len(X.index))
        Xd = X.copy()
        rows = len(Xd.index)
        if(self.fit_intercept == True):
            Xd.insert(0, "Intercept", pd.Series([1]*rows))
        col = list(Xd.columns)
        yd = y.copy()

        theta = np.array([1.0]*len(list(Xd.columns)))
        r = 0
        # self.thetas = []
        coef = []
        for i in range(n_iter):
            # self.thetas.append(theta)
            # print(theta)
            coef.append(list(theta))
            # print(coef)
            xx = pd.DataFrame(columns=col)
            yy = pd.Series([])
            for j in range(batch_size):
                xx.loc[j] = Xd.loc[r%rows]
                yy[j] = yd[r%rows]
                r += 1
            if (lr_type == 'inverse') and i!=0:
                lr = lr*i/(i+1)
            gradient = []
            for ii in range(len(col)):
                cols = [t for t in range(len(col))]
                xx.columns = cols
                der = 0
                # print(xx)
                for jj in range(batch_size):
                    der += 2*(yy[jj] - (np.array(xx.loc[jj]) @ np.array(theta)))*(-1*xx.loc[jj][ii]) #(self.error(xx.loc[jj], yy[jj], theta))
                der = der/batch_size
                gradient.append(der)
            for k in range(len(theta)):
                theta[k] -= (lr * gradient[k])
            # print(gradient)
        # print(self.thetas)
        # print(coef)
        self.thetas = coef
        # print(self.thetas)
        self.coef_ = pd.Series(theta)
        return self.coef_

    # def error(self, x_single, y_single, theta):
    #     y_hat = 0
    #     for i in range(x_single.size):
    #         y_hat += x_single[i] * theta[i]
    #     return (y_single - y_hat)


    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=1, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        assert (batch_size <= len(X.index))
        Xd = X.copy()
        yd = y.copy()
        rows = len(Xd.index)
        if(self.fit_intercept == True):
            Xd.insert(0, "Intercept", pd.Series([1]*rows))
        col = list(Xd.columns)
        theta = np.array([1.0]*len(col))
        r = 0
        coef = []
        for i in range(n_iter):
            coef.append(list(theta))
            xx = pd.DataFrame(columns=list(col))
            yy = pd.Series([])
            for j in range(batch_size):
                xx.loc[j] = Xd.loc[r%rows]
                yy[j] = yd[r%rows]
                r += 1
            if (lr_type == 'inverse') and i!=0:
                lr = lr*i/(i+1)
            Xn = xx.to_numpy()
            Xt = Xn.transpose()
            yn = yy.to_numpy()
            theta = theta - (Xt @ ((Xn @ theta) - yn))*(lr/batch_size)
        self.thetas = coef
        self.coef_ = pd.Series(theta)
    
    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        assert (batch_size <= len(X.index))
        rows = len(X.index)
        lr = lr
        Xd = X.copy()
        yd = y.copy()
        if(self.fit_intercept==True):
            Xd.insert(0,"Intercept",pd.Series([1]*rows))
        col = list(Xd.columns)
        theta = np.array([1.0]*len(col))
        r = 0
        coef = []
        for i in range(n_iter):
            coef.append(list(theta))
            xx = pd.DataFrame(columns=col)
            yy = pd.Series([])
            for j in range(batch_size):
                xx.loc[j] = list(Xd.loc[r%rows])
                yy[j] = yd[r%len(Xd.index)]
                r += 1
            self.xd = xx.to_numpy()
            self.ydd = yy.to_numpy()
            if (lr_type == 'inverse') and i!=0:
                lr = lr*i/(i+1)
            gradd = grad(self.error)
            gradient = gradd(theta)
            theta -= gradient * lr
        self.thetas = coef
        self.coef_ = pd.Series(theta)
            

    def error(self, theta):
        y_hat = self.xd.dot(theta)
        err = 0
        for i in range(len(self.ydd)):
            err += (y_hat[i] - self.ydd[i])**2
        return (err/(len(self.ydd)))

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        X = X.to_numpy()
        if self.fit_intercept:
            a = np.ones((X.shape[0],1))
            X = np.concatenate((a,X), axis=1)
        inv = np.linalg.pinv(X.T @ X)
        self.coef_ = inv @ X.T @ y
        return self.coef_

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X = X.to_numpy()
        if self.fit_intercept:
            a = np.ones((X.shape[0],1))
            X = np.concatenate((a,X), axis=1)
        self.prediction = X @ self.coef_
        return self.prediction 

    def plot_surface(self, X, y, t_0=1, t_1=1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        theta0 = np.linspace(-5,5,50)
        theta1 = np.linspace(-5,5,50)
        error = []
        r = y.size
        for i in range(len(theta0)):
            t0 = theta0[i]
            e = []
            for j in range(len(theta1)):
                t1 = theta1[j]
                err=0
                for k in range(r):
                    # print(y[k], t0+(t1*X[k]))
                    err+=float((y[k] - (t0+(t1*X[k])))**2)
                e.append(err/r)
            error.append(e)
        error = np.array(error)
        # print(error)
        theta_0 = np.outer(theta1, np.ones(50))
        theta_1 = theta_0.T
        fig = plt.figure()
        plt.ion()
        ax = plt.axes(projection = '3d')
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_title('RSS')
        ax.set_xlabel(r'$\Theta_{0}$')
        ax.set_ylabel(r'$\Theta_{1}$')
        ax.plot_surface(theta_0, theta_1, error, cmap = 'viridis', edgecolor = 'none', alpha =0.7)
        thetas = np.array(self.thetas)
        # print(self.thetas)
        er = 0
        for i in range(r):
            er+= float((y[i] - (t_0+(t_1*X[i])))**2)
        er = er/r
        eror = np.array([er])
        sc = ax.scatter(thetas[0:1, 0], thetas[0:1, 1], eror[0:1])
        fig.show()
        tet = []
        ind = np.linspace(0, r-1, 10)
        for i in range(10):
            tet.append(self.thetas[int(ind[i])])
        teta = np.array(tet)
        ero = []
        for i in range(10):
            plt.pause(0.5)
            er=0
            for k in range(r):
                er+=float((y[k] - (teta[i:i+1,0]+(teta[i:i+1,1]*X[k])))**2)
            er = er/r
            ero.append(er)
            eror = np.array(ero)
            sc._offsets3d = (teta[0:i+1,0], teta[0:i+1,1], eror[0:i+1])
            plt.draw()


    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """

        fig = plt.figure()
        x = X.to_numpy()
        p = np.linspace(min(x)-1,max(x)+1,(max(x)-min(x))*10)
        l = t_0 + t_1*p
        plt.plot(p,l,'-r', label='y=t_0+t_1*x)')
        plt.legend(loc='upper left')
        ax.scatter(x,y)
        plt.show()
        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as featuresauto (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        pass
