import tensorflow as tf
import pandas as pd
import numpy as np


data=pd.read_csv("NewData.csv",header=None)
data_x=data.iloc[1:,0:].values
data_y=data.iloc[1:,0].values

test=data_x[1]

x_data = data_x.astype(np.float32)
x_target = data_y.astype(np.float32)

model=tf.keras.models.load_model("mymodel.h5")

res=model.predict(test)

print(res)