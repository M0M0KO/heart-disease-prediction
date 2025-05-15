import numpy as np
import pandas as pd

data=pd.read_csv("heart_2020_cleaned.csv")
data_new=[]
for i in data:
    data_new.append(data[i])
# print(data_new)
for line in data_new:
    # print(line)
    for item in line:
        # print(item)
        if item=='Yes' or item=='Male':
            item=1
        elif item=='No' or item=='Female':
            item=0
        elif item =='Very good':
            item=5
        elif item =='Good':
            item=4
        elif item =='Excellent':
            item=3
        elif item =='Fair':
            item=2
        elif item =='Poor':
            item=1

for i in range(1,len(data_new[9])):
    ch=data_new[9][i]
    data_new[9][i]=ch[0:2]
    

print(data_new[0])
print(data_new[9])
