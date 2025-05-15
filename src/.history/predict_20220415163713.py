import tensorflow as tf
import pandas as pd
import numpy as np
import json
import tensorflow.keras.models as mod
#这是独立的包
# import tensorflow.keras

data=pd.read_csv("NewData.csv",header=None)
data_x=data.iloc[1:,0:].values
data_y=data.iloc[1:,0].values

test=data_x[1]

x_data = data_x.astype(np.float32)
x_target = data_y.astype(np.float32)

model=mod.load_model('mymodel.h5',custom_objects={'prediction':tf.keras.losses.categorical_crossentropy,'label':my_MSE})

res=model.prediction(test)

print(res)