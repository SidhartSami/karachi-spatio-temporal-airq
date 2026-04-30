import pandas as pd
import os

# Check both data directories
paths = [
    'data/processed/master_dataset.csv',
    'notebooks/data/processed/master_dataset.csv'
]

for path in paths:
    print(f"\n=== Checking: {path} ===")
    print(f"File exists: {os.path.exists(path)}")
    
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"Shape: {df.shape}")
            print(f"PM2.5 column: {'pm25' in df.columns}")
            if 'pm25' in df.columns:
                print(f"PM2.5 range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f}")
        except Exception as e:
            print(f"Error loading: {e}")
    else:
        print("File not found")
