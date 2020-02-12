import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

        pass
    
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
        if(self.fit_intercept == True):
            Xd.insert(0, "Intercept", pd.Series([1]*len(Xd.index)))
        col = list(Xd.columns)
        yd = y.copy()

        theta = np.array([1.0]*len(list(Xd.columns)))
        count = 0

        for i in range(n_iter):
            x = pd.DataFrame(columns=col)
            y_dash = pd.Series([])
            for j in range(batch_size):
                x.loc[j] = Xd.loc[count%len(Xd.index)]
                y_dash[j] = yd[count%len(Xd.index)]
                count += 1
            if (lr_type == 'inverse'):
                lr = lr/(i+1)
            gradient = []
            for ii in range(len(col)):
                cols = [t for t in range(len(col))]
                x.columns = cols
                der = 0
                # print(x)
                for jj in range(batch_size):
                    der += 2*(self.error(x.loc[jj], y_dash[jj], theta))*(-1*x.loc[jj][ii])
                der = der/batch_size
                gradient.append(der)
            for k in range(len(theta)):
                theta[k] -= (lr * gradient[k])
        self.coef_ = pd.Series(theta)

    def error(self, x_single, y_single, theta):
        y_hat = 0
        for i in range(x_single.size):
            y_hat += x_single[i] * theta[i]
        return (y_single - y_hat)


    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
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
        if(self.fit_intercept == True):
            Xd.insert(0, "Intercept", pd.Series([1]*len(Xd.index)))
        col = list(Xd.columns)
        theta = np.array([1.0]*len(col))
        count = 0
        for i in range(n_iter):
            x = pd.DataFrame(columns=list(col))
            y_dash = pd.Series([])
            for j in range(batch_size):
                x.loc[j] = Xd.loc[count%len(Xd.index)]
                y_dash[j] = yd[count%len(Xd.index)]
                count += 1
            if (lr_type == 'inverse'):
                lr = lr/(i+1)
            Xn = x.to_numpy()
            Xt = Xn.transpose()
            yn = y_dash.to_numpy()
            theta = theta - (Xt @ ((Xn @ theta) - yn))*(lr/batch_size)
        print(theta)
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

        pass

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
        self.theta = inv @ X.T @ y
        return self.theta

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

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """

        pass

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

        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        pass
