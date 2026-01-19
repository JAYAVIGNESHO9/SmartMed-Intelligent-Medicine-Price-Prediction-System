import pandas as pd

df = pd.read_csv("D:\Project\learnML\Smart Medicine Price & Availability Predictor\data\indian_pharmaceutical_products.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.columns.tolist())