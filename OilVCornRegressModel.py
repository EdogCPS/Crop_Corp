import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("TimeVOilVCorn.csv")
y = data["oil price"]
x = data["monthsFrom2014"]
fig, graph = plt.subplots(1)
graph.scatter(x, y)
graph.set_xlabel("Months from 2014")
graph.set_ylabel("oil price")
plt.tight_layout()
plt.show()

