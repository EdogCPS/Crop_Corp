import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("TimeVOilVCorn.csv")
y_1 = data["oil price"]
y_2 = data["corn price"]
x = data["Time"]
# x_data=StandardScaler().fit_transform(x)
# y_data=StandardScaler().fit_transform(y)
fig, graph = plt.subplots(2)
graph[0].scatter(x, y_1)
graph[0].set_xlabel("Time")
graph[0].set_ylabel("oil price ")

graph[1].scatter(x, y_2)
graph[1].set_xlabel("Time")
graph[1].set_ylabel("corn price")

plt.tight_layout()
plt.show()

