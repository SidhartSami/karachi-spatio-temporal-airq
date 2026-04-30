import pandas as pd
import os

# Check file exists and its contents
file_path = 'data/processed/master_dataset.csv'
print(f"File exists: {os.path.exists(file_path)}")

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"File shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"PM2.5 in columns (lowercase): {'pm25' in df.columns}")
    print(f"PM2.5 in columns (uppercase): {'PM25' in df.columns}")
    print(f"PM2.5 in columns (mixed): {'PM2.5' in df.columns}")
    
    # Show first few rows
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    # Check if pm25 column has data
    if 'pm25' in df.columns:
        print(f"\nPM2.5 stats: {df['pm25'].describe()}")
else:
    print("File does not exist!")
