import numpy as np
import pandas as pd

data=pd.read_csv("heart_2020_cleaned.csv")
data_new=[]
for i in data:
    data_new.append(data[i])
news=[]
# print(data_new)
for line in data_new:
    i=0
    news.append()
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
        news[i].append(item)
    i=i+1

for i in range(1,len(news[9])):
    ch=news[9][i]
    news[9][i]=ch[0:2]
    

print(news[0])
print(news[9])
