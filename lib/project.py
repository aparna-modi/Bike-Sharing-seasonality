import pandas as pd
import numpy as np 
from statsmodels.tsa.seasonal import STL 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit 



df = pd.read_csv("./data/day.csv") 

# 2. Convert 'dteday' to a proper datetime
df['dteday'] = pd.to_datetime(df['dteday'])
df.set_index('dteday', inplace=True) 

df['cnt'] = df['cnt'].interpolate(method='linear')
print("Data loaded and validated!")

df.to_csv("./data/cleaned.csv", index_label='dteday')

stl = STL(df['cnt'], seasonal=7) # 7-day weekly seasonality 
result = stl.fit() 

# Plot the components 
fig = result.plot() 
plt.show() 
plt.savefig('stl.png')

# Extract the residual noise 
residuals = result.resid



# Baseline 1: Naive Forecast (Shift data by 1 day) 
df['Naive_Forecast'] = df['cnt'].shift(1) 

# Baseline 2: 7-Day Moving Average 
df['7D_Moving_Avg'] = df['cnt'].shift(1).rolling(window=7).mean() 
# Drop NaNs created by shifting 
eval_df = df.dropna() 

mape_naive = mean_absolute_percentage_error(eval_df['cnt'], eval_df['Naive_Forecast']) 

mape_ma = mean_absolute_percentage_error(eval_df['cnt'], eval_df['7D_Moving_Avg']) 

print(f"Naive Forecast MAPE: {mape_naive:.2%}") 
print(f"Moving Average MAPE: {mape_ma:.2%}")



# Create 5 sliding windows for cross-validation 
tscv = TimeSeriesSplit(n_splits=5) 

for train_index, test_index in tscv.split(eval_df): 
	train, test = eval_df.iloc[train_index], eval_df.iloc[test_index] 