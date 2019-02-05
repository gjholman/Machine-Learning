import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# import csv of honey production using pandas
df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

# Pandas function head will show the first 5 rows of data.
#print(df.head())

# We now know from printing that we have 8 rows:
# state, numcol, uorldpercol, totalprod, stocks, priceperlb, prodvalue, year
# Each row represents a state, and in the csv they are organized alphabetically

# Use pandas to group by year and get the mean of the total prod for each year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# Now the data is grouped by year with the average total production for each state

# Create variable X that is the values in the years column
X = prod_per_year['year']

# Reshape into the right format for sci-kit
# Essentially it takes an array and makes it a column. Rotates the array
X = X.values.reshape(-1, 1)

# Create variable y that is the totalprod column in thea
y = prod_per_year['totalprod']

# Use matplotlib (as plt) to show a scatter plot
plt.scatter(X, y, label="Honey Production Data")

# Now that we have a plot of the data, we can see that there is a linear
# tendancy, which we can determine using scikit's Linear Regression Model
regr = linear_model.LinearRegression()

# Now we fit the data to the Linear Regression
regr.fit(X, y)

#Print out the values of the coefficient and the intercept
#print(regr.coef_)
#print(regr.intercept_)

# List of predictions for the X values by the regr Linear Regression
y_predict = regr.predict(X)

# Plot the X values and the y-predict
plt.plot(X, y_predict, 'r', label="Linear Regression Model")
#plt.show()

#================= Predict Honey Decline ===============
# Let's predict honey decline in 2050. 

# Create a numpy array in the range of 2013-2050
X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1)

# Predict the future using the regr linear regression for the future X values
y_future_predict = regr.predict(X_future)
plt.plot(X_future, y_future_predict, 'g', label="Future Prediction")

plt.title('Honey Production')
plt.xlabel('Year')
plt.ylabel('Average Honey Production')
plt.legend()
plt.show()

print("By the year 2050, the average honey production value will decline to: {0}".format(regr.predict(2050).round()))
