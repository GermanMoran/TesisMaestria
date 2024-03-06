from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression


X, y = make_regression(n_features=2, random_state=0)
print(X, print)
regr = ElasticNet(random_state=0)
regr.fit(X, y)
print(regr.coef_)
print(regr.intercept_)
print(regr.predict([[0, 0]]))