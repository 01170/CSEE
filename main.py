#This is my CSEE project

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

num_of_days = 7

#Store data in variable
df = pd.read_csv('TSLA.csv', usecols=['Date', 'Open', 'Close'])

# Read column names from file
cols = list(pd.read_csv("TSLA.csv", nrows =1))
print(cols)

#Set date as index
df = df.set_index(pd.DatetimeIndex(df['Date'].values))

#Name index
df.index.name = 'Date'

# Initialize the regressors (x) and the predictors (y).
x = np.array([])
y = np.array([])

# Iterate over the data setting the regressors as the open values
# and the predictors as the closing values.
for i, j in df.iterrows():
  x = np.append(x, j['Open'])
  y = np.append(y, j['Close'])

# Setup the model
x = x.reshape((-1, 1))
model = LinearRegression()
model.fit(x, y)

# Print some useful info
print('coefficient of determination:', model.score(x, y))
print('predicted response for 200:', model.predict([[200]]))
print('slope:', model.coef_)

# Build out some number of days using previous close values as new open values
new_open = model.predict([[df.iloc[len(df) - 1]['Close']]])
z = np.array([])
for x in range(num_of_days):
  p = model.predict([new_open])
  new_open = p
  z = np.append(z, p)

#Create target column
df['PriceUp'] = np.where(df['Close'].shift(-1) > df["Close"], 1, 0)

#Remove additional Date column
df = df.drop(columns=['Date'])
print(df)
print(z)