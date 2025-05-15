import numpy as np
import pandas as pd

data=pd.read_csv("heart_2020_cleaned.csv")
data_new=data
for line in data_new:
    print(line)
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

# for item in data_new['AgeCategory']:
#     item=item[0:2]

# print(data_new['HeartDisease'])
# print(data_new['AgeCategory'])
