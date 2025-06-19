#importing libraries
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Load the dataset
df = pd.read_csv("water-quality.csv", sep=";")
print(df)

#data preprocessing
print(df.info())
print(df.shape)
print(df.describe().T)

print(df.isnull().sum())

#converting to date-format
df["date"] = pd.to_datetime(df["date"],format='%d.%m.%Y')

#sorting the rows based on date
df = df.sort_values(by=['id', 'date'])

#adding the columns month and year 
df["month"]=df["date"].dt.month
df["year"] = df["date"].dt.year
print(df.head())

print(df.columns)