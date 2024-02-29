import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("TimeVOilVCorn.csv")
x = data["oil price"].values
y = data["corn price"].values
x = x.reshape(-1, 1)
model = LinearRegression().fit(x,y)
fig, graph = plt.subplots(1)
graph.scatter(x, y)
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)
x_predict=83.2    
graph.set_xlabel("corn price")
graph.set_xlabel("oil price")
prediction = model.predict([[x_predict]])
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")
plt.legend
plt.tight_layout()
print(f"Corn price prediction when oil is ${x_predict}: ${prediction}")
plt.show

