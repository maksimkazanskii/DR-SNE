import pandas as pd
df = pd.read_csv("data/cleaned_dataset_Thyroid1.csv")

print(df.shape)
print(df.columns)
print(df.head())