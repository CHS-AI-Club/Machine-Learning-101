import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import os, pickle

# this part can be ignored; just code for getting current working directory
cwd = os.path.dirname(os.path.abspath(__file__))
train_set_path = os.path.join(cwd, "train_set.csv")
test_set_path = os.path.join(cwd, "test_set.csv")

# load in the data and separate training variables
train_set = pd.read_csv(train_set_path, index_col=0)
test_set = pd.read_csv(test_set_path, index_col=0)
x_train = train_set["distance"].values.reshape(-1, 1)
y_train = train_set["bullet_drop"].values.reshape(-1, 1)

# make the model and fit/train the model with the data
poly_model = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression())
poly_model.fit(x_train, y_train)

# creates a scatter plot with a regression curve and saves it
yfit = poly_model.predict(x_train)
plt.scatter(x_train, y_train)
plt.plot(x_train, yfit, color="red", label="Regression Curve")
plt.legend()
plt.xlabel("Horizontal Distance (meters)")
plt.ylabel("Bullet Drop (meters)")
figure_path = os.path.join(cwd, "graph.png")
plt.savefig(figure_path)
plt.show()

# saves model to directory
model_path = os.path.join(cwd, "poly_model.pickle")
with open(model_path, "wb") as f:
    pickle.dump(poly_model, f)

# loads back the model into the file as to show that it works
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

# predict bullet drop after 150 meters, comparison with the actual value  
prediction = loaded_model.predict([[150]])
prediction = round(prediction[0][0], 4)
actual_value = test_set.loc[test_set["distance"] == 150]
actual_value = round(float(actual_value['bullet_drop']), 4)
difference = round(abs(actual_value-prediction), 4)
error = round((difference / abs(actual_value)) * 100, 4)  # may not be accurate due to rounding errors
print(f"Prediction of the bullet drop at 150 meters: {prediction} meters")
print(f"Actual bullet drop at 150 meters: {actual_value} meters")
print(f"Difference: {difference}")
print(f"% Error: {error}")