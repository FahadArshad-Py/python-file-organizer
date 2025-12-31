import pandas as pd
import math

df = pd.read_csv("Data.csv")
df.insert(4, 'HI', 0.0)
df.insert(5, 'RL', 0.0)
df.insert(6, 'REM', '')

BM = float(input("Enter BM for the first point: "))
df.loc[0, 'RL'] = round(BM, 3)
df.loc[0, 'REM'] = 'BM'

for i in range(0,len(df)):
    if df.loc[i, 'BS'] != 0:
        df.loc[i, 'HI'] = df.loc[i, 'RL'] + df.loc[i, 'BS']
    else:
         df.loc[i, 'HI'] = df.loc[i-1, 'HI']
         
    if df.loc[i, 'IS'] != 0:
        df.loc[i, 'RL'] = df.loc[i, 'HI'] - df.loc[i, 'IS']

    elif df.loc[i, 'FS'] != 0:
        df.loc[i, 'HI'] = df.loc[i-1, 'HI']
        df.loc[i, 'RL'] = df.loc[i, 'HI'] - df.loc[i, 'FS']
        df.loc[i,'HI'] = df.loc[i,'RL'] + df.loc[i,'BS']
        df.loc[i, 'REM'] = 'CP'
    else:
         df.loc[i, 'HI'] = df.loc[i, 'HI']

sumbs = df['BS'].sum()
sumfs = df['FS'].sum()
lastvalueRL = df.loc[len(df)-1,'RL']
firstvalueRL = df.loc[0,'RL']

check1 = "sumbs - sumfs = " + str(round(sumbs - sumfs,3))
check2 = "Last RL - First RL = " + str(round(lastvalueRL - firstvalueRL,3))

print(df[['point', 'BS', 'IS', 'FS', 'HI', 'RL', 'REM']])
print()
print("checks")
print(check1)
print(check2)