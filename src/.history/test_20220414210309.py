import pandas as pd 

print('引用完成')

data=pd.read_csv("NewData.csv")
data_x=data.iloc[1:,1:]
data_y=data.iloc[1:,0]

print(data_x)