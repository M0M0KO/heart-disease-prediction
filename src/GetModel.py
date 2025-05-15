from gc import callbacks
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
# from sklearn.atasets import load_iris
# data = load_iris()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


data=pd.read_csv("NewData.csv",header=None)
data_x=data.iloc[1:,0:].values
data_y=data.iloc[1:,0].values

x_data = data_x.astype(np.float32)
x_target = data_y.astype(np.float32)
# iris_data = np.float32(data.data)
# iris_data_1 = []
# iris_data_2 = []

# for iris in iris_data:
#     iris_data_1.append(iris[:2])
#     iris_data_2.append(iris[2:])
iris_label = np.float32(data_y)
iris_target = np.float32(tf.keras.utils.to_categorical(x_target,num_classes=2))
train_data = tf.data.Dataset.from_tensor_slices(((x_data,x_data),(iris_target,iris_label))).batch(32)
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
history=LossHistory()
model.compile(optimizer=opt,loss={'prediction':tf.keras.losses.categorical_crossentropy,'label':my_MSE},loss_weights={'prediction':0.1,'label':0.5},metrics=['accuracy','mse'])
model.fit(x=train_data,epochs=20,callbacks=[history])
score = model.evaluate(train_data)
history.loss_plot('epoch')
# model.save('mymodel.h5')

print('last score is :',score)
