#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

print(df.columns)

pollutants = ['O2', 'NO3', 'NO2', 'SO4','PO4', 'CL']

#Removing the null values
df=df.dropna(subset=pollutants)
print(df.head())

print(df.isnull().sum())

#Line Plot for Pollutants Over Multiple Years
pollutant_means_by_year = df.groupby('year')[pollutants].mean().reset_index()
melted_pollutants = pollutant_means_by_year.melt(id_vars='year', var_name='Pollutant', value_name='Average Value')
plt.figure(figsize=(14, 6))
sns.lineplot(data=melted_pollutants, x='year', y='Average Value', hue='Pollutant', marker='o', palette='tab10')

plt.title("Yearly Average Pollutant Levels Over Time", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Average Pollutant Concentration")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Feature and target selection
X=df[['id','year']]
Y=df[pollutants]

# Encoding - onehotencoder 
X_encoded=pd.get_dummies(X,columns=['id'],drop_first=True)

# Train, Test and Split
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded,Y,test_size=0.2,random_state=42)

#Training the model
model=MultiOutputRegressor(RandomForestRegressor(n_estimators=100,random_state=42))
model.fit(X_train,Y_train)

# Evaluate model
y_pred = model.predict(X_test)

#MSE: how far off the model is from actual values (lower is better)
# #R² score: how well the model explains variance in the data (higher is better)
print("Model Performance on the Test Data:")
for i, pollutant in enumerate(pollutants):
    print(f'{pollutant}:')
    print('MSE:', mean_squared_error(Y_test.iloc[:, i], y_pred[:, i]))
    print('R2:', r2_score(Y_test.iloc[:, i], y_pred[:, i]))
    print()

#Predict the pollutants in a specific station   
station_id = '5'
year_input = 2024

input_data = pd.DataFrame({'year': [year_input], 'id': [station_id]})
input_encoded = pd.get_dummies(input_data, columns=['id'])
# Align with training feature columns
missing_cols = set(X_encoded.columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]  # reorder columns

predicted_pollutants = model.predict(input_encoded)[0]

print(f"\nPredicted pollutant levels for station '{station_id}' in {year_input}:")
for p, val in zip(pollutants, predicted_pollutants):
    print(f"  {p}: {val:.2f}")

#  Bar Plot – Compare All Predicted Pollutants
safety_thresholds = {
    'O2': 5.0,     
    'NO3': 10,
    'NO2': 0.1,
    'SO4': 250,
    'PO4': 0.1,
    'CL': 250
}
prediction_df = pd.DataFrame({
    'Pollutant': pollutants,
    'Predicted Value': predicted_pollutants
})
plt.figure(figsize=(10, 6))
sns.barplot(data=prediction_df, x='Pollutant', y='Predicted Value', palette='coolwarm')

for i, pollutant in enumerate(prediction_df['Pollutant']):
    predicted_val = predicted_pollutants[i]
    threshold = safety_thresholds[pollutant]
    if pollutant=='O2' and predicted_val<threshold:
        plt.text(i, threshold + 0.5, f"Limit: {threshold}", ha='center', color='red', fontsize=8)       
    elif pollutant!='O2' and predicted_val > threshold:
        plt.text(i, threshold + 0.5, f"Limit:<{threshold}", ha='center', color='red', fontsize=8)
       
plt.title(f"Predicted Pollutant Levels for Station {station_id} in {year_input}", fontsize=14)
plt.xlabel("Pollutants")
plt.ylabel("Predicted Concentration")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import joblib

joblib.dump(model, 'pollution_model.pkl')
joblib.dump(X_encoded.columns.tolist(), "model_columns.pkl")
print('Model and cols structure are saved!')
