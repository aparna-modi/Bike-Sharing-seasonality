from statsmodels.tsa.stattools import adfuller 

def adfuller_test(cnt):
    result = adfuller(cnt)
    labels = ['ADF test statistics', 'P-value', '#Lags used', 'Number of observation used']
    for value, label in zip(result, labels):
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print('Data is stationary.')
    else:
        print('Data is non stationary.')
        
        
adfuller_test(df['cnt'])
