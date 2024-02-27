import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("TimeVOilVCorn.csv")
y = data["corn price"]
x = data["monthsFrom2014"]
fig, graph = plt.subplots(1)
graph.scatter(x, y)
graph.set_xlabel("months from 2014")
graph.set_ylabel("corn price")
plt.tight_layout()
plt.show()
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=.2)
log_reg=linear_model.LogisticRegression()
log_reg.fit(xtrain,ytrain)
print(log_reg.score(xtest,ytest))