import pandas as pd

# Read the dataset
df = pd.read_csv('data/dataset/ddos_dataset.csv')

# Print column names
print("\nColumns in the dataset:")
for col in df.columns:
    print(f"'{col}',")
