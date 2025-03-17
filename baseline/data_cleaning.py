import pandas as pd
# import numpy as np

input_file = 'baseline_data_raw.csv'
output_file = 'baseline_data_cleaned.csv'

df = pd.read_csv(input_file)

print("raw data: ")
print(df.head())
print("\nraw data overview:")
print(df.info())

# housing: 'yes' -> 1, 'no' -> 0
df['housing'] = df['housing'].map({'yes': 1, 'no': 0})

# y: 'yes' -> 1, 'no' -> 0
df['y'] = df['y'].map({'yes': 1, 'no': 0})

print("\ncleaned data:")
print(df.head())

df.to_csv(output_file, index=False)