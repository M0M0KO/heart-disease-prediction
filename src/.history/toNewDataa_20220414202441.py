import numpy as np
import pandas as pd
import csv

data=pd.read_csv("heart_2020_cleaned.csv")
data_new=[]
data_head=[i for i in data]
for i in data:
    data_new.append(data[i])
news=[]
# print(data_new)
for line in data_new:
    new=[]
    # print(line)
    for item in line:
        # print(item)
        if item=='Yes' or item=='Male':
            item=1
        elif item=='No' or item=='Female':
            item=0
        elif item =='Very good' or item=='American Indian/Alaskan Native':
            item=5
        elif item =='Good' or item=='Asian':
            item=4
        elif item =='Excellent' or item=='Black':
            item=3
        elif item =='Fair' or item=='Hispanic':
            item=2
        elif item =='Poor' or item=='Other':
            item=1
        elif item =='White':
            item=6
        new.append(item)
    news.append(new)
# print(news[6])
for i in range(1,len(news[9])):
    ch=news[9][i]
    # print(ch)
    news[9][i]=ch[0:2]

print(data_head)
    

