import pandas as pd

melbourne_file_path = "melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0) #this is to remove the missing values from the datasets

#Selecting the Prediction Target
y = melbourne_data.Price


#Choosing the "Features"
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.head())


#Building My Model
from sklearn.tree import DecisionTreeRegressor

#Define model
melbourne_model = DecisionTreeRegressor(random_state=1)

#Fit the Model
melbourne_model.fit(X, y)

#Predictions
print("Making predictions for the following 5 homes:")
print(X.head())
print("The Predictions are")
print(melbourne_model.predict(X.head()))
print("No")