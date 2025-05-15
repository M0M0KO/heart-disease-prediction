from distutils.sysconfig import customize_compiler
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import tensorflow.keras.models as mod
#这是独立的包
import tensorflow.keras

data=pd.read_csv("NewData.csv",header=None)
data_x=data.iloc[1:,0:].values
data_y=data.iloc[1:,0].values

test=data_x[1]
test=test.astype(np.float32)
# x_data = data_x.astype(np.float32)
# x_target = data_y.astype(np.float32)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
def my_MSE(y_ture,y_pred):
    my_loss = tf.reduce_mean(tf.square(y_ture-y_pred))
    return my_loss

x_data = data_x.astype(np.float32)
x_target = data_y.astype(np.float32)
data_x=data.iloc[1:,0:].values
data_y=data.iloc[1:,0].values

iris_label = np.float32(data_y)
iris_target = np.float32(tf.keras.utils.to_categorical(x_target,num_classes=2))

x_data = data_x.astype(np.float32)
train_data = tf.data.Dataset.from_tensor_slices(((x_data,x_data),(iris_target,iris_label))).batch(32)


model=tf.keras.models.load_model('mymodel.h5',compile=False)
model.compile(optimizer=opt,loss={'prediction':tf.keras.losses.categorical_crossentropy,'label':my_MSE},loss_weights={'prediction':0.1,'label':0.5},metrics=['accuracy','mse'])


res=model.predict(train_data)
res=np.argmax(res[0], axis=1)
print(res[0][0])