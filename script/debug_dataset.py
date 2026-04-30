import pandas as pd

# Check what the notebook is actually loading
df = pd.read_csv('data/processed/master_dataset.csv')
print('Master dataset columns:')
print(list(df.columns))
print(f'PM25 in columns: {"pm25" in df.columns}')

# Check if notebook might be loading wrong file
try:
    df_old = pd.read_csv('data/processed/merged_karachi_dataset.csv')
    print(f'\nOld dataset columns: {list(df_old.columns)}')
    print(f'PM25 in old dataset: {"pm25" in df_old.columns}')
except:
    print('\nOld dataset not accessible')
