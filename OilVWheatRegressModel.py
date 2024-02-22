import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("2._Energy_Prices.csv")

x = data["Date"].values
y = data["Price"].values

x = x.reshape(-1, 1)

model = LinearRegression().fit(x,y)

coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

x_predict = 2012024

prediction = model.predict([[x_predict]])
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")
print(f"Prediction when x is {x_predict}: {prediction}")

plt.figure(figsize=(6,4))
plt.scatter(x,y, c ="purple")
plt.scatter(x_predict, prediction, c="blue")

plt.xlabel("Dates (Monthly)")
plt.ylabel("Prices of Corn Futures")
plt.title("Monthly Prices of Corn Futures")
plt.xticks(np.arange(2014, 2025, step=1))
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.legend
plt.show()