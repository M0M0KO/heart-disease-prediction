import tensorflow as tf
import numpy as np
import pandas as pd

input_xs_1 = tf.keras.Input(shape=(18),name='input_xs_1')
input_xs_2 = tf.keras.Input(shape=(18),name='input_xs_2')
input_xs = tf.concat([input_xs_1,input_xs_2],axis=-1)
out = tf.keras.layers.Dense(32,activation='relu',name = 'dense_1')(input_xs)
out = tf.keras.layers.Dense(64,activation='relu',name='dense_2')(out)
logits = tf.keras.layers.Dense(2,activation='softmax',name='prediction')(out)  # out from prediction 1
label= tf.keras.layers.Dense(1,name='label')(out) # out from prediction 2
model = tf.keras.Model(inputs = [input_xs_1,input_xs_2],outputs = [logits,label])
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
def my_MSE(y_ture,y_pred):
    my_loss = tf.reduce_mean(tf.square(y_ture-y_pred))
    return my_loss
model.compile(optimizer=opt,loss={'prediction':tf.keras.losses.categorical_crossentropy,'label':my_MSE},loss_weights={'prediction':0.1,'label':0.5},metrics=['accuracy','mse'])
model.fit(x=train_data,epochs=20)