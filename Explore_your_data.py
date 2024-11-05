import pandas as pd

iowa_file_path = "train.csv"
home_data = pd.read_csv(iowa_file_path)

print(home_data.describe())