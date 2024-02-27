import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("TimeVOilVCorn.csv")
y = data["corn price"]
x_1 = data["time(year)"]
x_2 = data["time(month)"]
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=.2)
log_reg=linear_model.LogisticRegression()
log_reg.fit(xtrain,ytrain)
print(log_reg.score(xtest,ytest))