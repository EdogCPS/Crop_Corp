import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# gets the data and sets x and y values
data = pd.read_csv("TempVCorn.csv")
x = data["Temp"].values
y = data["CornP"].values
print(data)
print(x)
print(y)

# shift the y-values to have the origin at 0
y_shifted = y - np.min(y)

# use reshape to turn the x values into a 2D array
x = x.reshape(-1, 1)
print(x)

# create the model
model = LinearRegression().fit(x, y_shifted) # use shifted y-values

# find the coefficient, bias, and r squared values
# each should be a float and rounded to two decimal places
#model.coef (is the slope of the graph)
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y_shifted) # use shifted y-values
print(coef, intercept, r_squared)

# value you are going to predict
x_predict = 63
# plug that value into your model
prediction_shifted = model.predict([[x_predict]])

# undo the y-value shift for prediction
prediction = prediction_shifted + np.min(y)

# print out the linear equation and r squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept + np.min(y)}")
print(f"R Squared value: {r_squared}")
print(f"Prediction when x is {x_predict}: {prediction}")

'''
The following code creates the graph to visualize the data
'''
# sets the size of the graph
plt.figure(figsize=(6,4))

# creates a scatter plot of original data in purple
# and the predicted data in blue
plt.scatter(x, y, c="purple")
plt.scatter(x_predict, prediction, c="blue")

# label the axes
plt.xlabel("Temperature °F")
plt.ylabel("Corn Prices")
plt.title("Corn Prices by Temperature")

# plot the line of best fit in red and label the line
plt.plot(x, coef*x + intercept + np.min(y), c="r", label="Line of Best Fit")

# show the plot and legend
plt.legend()
plt.show()