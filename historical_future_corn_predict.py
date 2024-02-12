import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv("US Corn Futures Historical Data (1).csv")

x = data["Date"]
y = data["Price"]

model = LinearRegression().fit(x,y)
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

plt.figure(figsize=(6,4))
plt.scatter(x,y, c ="purple")
plt.xlabel("Dates (Monthly)")
plt.ylabel("Prices of Corn Futures")

plt.legend
plt.show()