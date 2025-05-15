import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
data = load_iris()

iris_data = np.float32(data.data)
iris_data_1 = []
iris_data_2 = []

for iris in iris_data:
    iris_data_1.append(iris[:2])
    iris_data_2.append(iris[2:])
iris_label = np.float32(data.target)
iris_target = np.float32(tf.keras.utils.to_categorical(data.target,num_classes=3))
train_data = tf.data.Dataset.from_tensor_slices(((iris_data_1,iris_data_2),(iris_target,iris_label))).batch(128)
input_xs_1 = tf.keras.Input(shape=(2),name='input_xs_1')
input_xs_2 = tf.keras.Input(shape=(2),name='input_xs_2')
input_xs = tf.concat([input_xs_1,input_xs_2],axis=-1)
out = tf.keras.layers.Dense(32,activation='relu',name = 'dense_1')(input_xs)
out = tf.keras.layers.Dense(64,activation='relu',name='dense_2')(out)
logits = tf.keras.layers.Dense(3,activation='softmax',name='prediction')(out)  # out from prediction 1
label= tf.keras.layers.Dense(1,name='label')(out) # out from prediction 2
model = tf.keras.Model(inputs = [input_xs_1,input_xs_2],outputs = [logits,label])
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
def my_MSE(y_ture,y_pred):
    my_loss = tf.reduce_mean(tf.square(y_ture-y_pred))
    return my_loss
model.compile(optimizer=opt,loss={'prediction':tf.keras.losses.categorical_crossentropy,'label':my_MSE},loss_weights={'prediction':0.1,'label':0.5},metrics=['accuracy','mse'])
model.fit(x=train_data,epochs=500)
score = model.evaluate(train_data)
print('last score is :',score)
————————————————
版权声明：本文为CSDN博主「浪漫的数据分析」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_43290383/article/details/121600247