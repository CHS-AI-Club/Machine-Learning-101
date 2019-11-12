import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import os

cwd = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(cwd, "train_set.csv")

train_set = pd.read_csv(path, index_col=0)
x_train = train_set["distance"].values.reshape(-1, 1)
y_train = train_set["bullet_drop"].values.reshape(-1, 1)

poly_model = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression())
poly_model.fit(x_train, y_train)
yfit = poly_model.predict(x_train)
plt.scatter(x_train, y_train)
plt.plot(x_train, yfit)
plt.show()

pred = poly_model.predict([[150]])
print(pred)