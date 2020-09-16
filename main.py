#This is my CSEE project

import pandas as pd
import pip
import sklearn as sklearn
import numpy as np
import tweepy

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from Tweet import Tweet

#Store data in variable
df = pd.read_csv('TSLA.csv', usecols=['Date', 'Open', 'Close'])

# Read column names from file
cols = list(pd.read_csv("TSLA.csv", nrows =1))
print(cols)

#Set date as index
df = df.set_index(pd.DatetimeIndex(df['Date'].values))

#Name index
df.index.name = 'Date'

#for loop
#for x in df:
    #print(df['Close'])
    #print(df['Open'])

#Create target column
df['Price_Up'] = np.where(df['Close'].shift(-1) > df["Close"], 1, 0)

#Remove additional Date column
df = df.drop(columns=['Date'])
print(df)

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()
#model.fit(x, y)
#r_sq = model.score(x, y)

#print('coefficient of determination:', r_sq)

#y_pred = model.predict(x)
#print('predicted response:', y_pred, sep='\n')