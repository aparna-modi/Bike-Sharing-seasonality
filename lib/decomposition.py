from statsmodels.tsa.seasonal import STL 
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data/cleaned.csv")

df['dteday'] = pd.to_datetime(df['dteday'])
df.set_index('dteday', inplace=True)

# stl = STL(df['cnt'], seasonal=7) # 7-day weekly seasonality 
# result = stl.fit() 
stl = STL(df['cnt'],seasonal = 13, period=7, robust=True)
result = stl.fit()

fig = result.plot()
fig.set_size_inches(10, 8)
plt.suptitle('STL Decomposition of Daily Bike Rentals', fontsize=16)
plt.tight_layout()

# Save the plot to your directory
plt.savefig('stl_decomposition.png')
print("Decomposition complete! Chart saved as 'stl_decomposition.png'.")
# Extract the residual noise 
trend_component = result.trend
seasonal_component = result.seasonal
residual_component = result.resid