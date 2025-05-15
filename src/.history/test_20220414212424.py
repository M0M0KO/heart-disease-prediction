import pandas as pd 

data=pd.read_csv("NewData.csv",header=None)
data_x=data.iloc[1:,1:].values
data_y=data.iloc[1:,0]
for j in range(len(data_x[0])):
    for i in range(len(data_x)):
        data_x[i][j]=float(data_x[i][j])

for i in range(len(data_y)):
    data_y[i]=float(data_y[i])
print(data_x)