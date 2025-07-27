import pandas as pd

# Load your Excel file
df = pd.read_excel("dataset/original_data/metadata.xlsx")

# Save as CSV
df.to_csv("dataset/original_data/metadata.csv", index=False)

print("✅ metadata.csv saved.")
