import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures

X = np.array([1,2])
poly = PolynomialFeatures(2)
df = poly.transform(X)
print(df)
