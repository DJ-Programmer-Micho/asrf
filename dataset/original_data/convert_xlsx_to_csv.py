import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config_loader import load_config

config = load_config()
xlsx_path = config["dataset"]["original_data_path_xlsx"]
csv_path = config["dataset"]["original_data_path_csv"]

# Load your Excel file
df = pd.read_excel(xlsx_path)

# Save as CSV
df.to_csv(csv_path, index=False)

print("✅ metadata.csv saved.")
