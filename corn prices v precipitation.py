import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data1 = pd.read_csv("US Corn Futures Historical Data (1).csv")
data2 =pd.read_csv("ChicagoPrecipitationMonthlyData2014-2024.csv")
print(data1)

x_1 =data2["Months"]
x_2 =data2["Precipitation average"]
x=[x_1,x_2]
y =data1["Price"]
print(x)
print(y)
fig, graph = plt.subplots(2)
graph[0].scatter(x_1, y)
graph[0].set_xlabel("Months since January 2014")
graph[0].set_ylabel("Corn Future")

graph[1].scatter(x_2, y)
graph[1].set_xlabel("Precipitation")
graph[1].set_ylabel("Corn Future")

print("Correlation between Months since January 2014 and Corn Futures:",round(x_1.corr(y),2))
print("Montly Average Precipitation and Corn Futures:",round(x_2.corr(y),2))

plt.tight_layout()
plt.show()
