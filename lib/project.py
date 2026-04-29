import pandas as pd

df = pd.read_csv("./data/day.csv") 

# 2. Convert 'dteday' to a proper datetime
df['dteday'] = pd.to_datetime(df['dteday'])


df.set_index('dteday', inplace=True)


df['cnt'] = df['cnt'].interpolate(method='linear')
print("Data loaded and validated!")

df.to_csv("./data/cleaned.csv", index_label='dteday')
