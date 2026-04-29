from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit 
import pandas as pd

df = pd.read_csv("./data/cleaned.csv")

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