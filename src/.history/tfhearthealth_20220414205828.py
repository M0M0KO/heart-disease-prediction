import tensorflow as tf
import numpy as np
import pandas as pd 

print('引用完成')

data=pd.read_csv("NewData.csv")
data_x=data.iloc[1:,1:]
data_y=data.iloc[1:,0]


# 导入数据，分别为输入特征和标签
x_data = data_x ## 存花萼、花瓣特征数据
y_data = data_y # 存对应种类
# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)
# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-300]
y_train = y_data[:-300]
x_test = x_data[-300:]
y_test = y_data[-300:]
# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
w1 = tf.Variable(tf.random.truncated_normal([17, 2], stddev=0.1)) # 四行三列,方差为0.1
b1 = tf.Variable(tf.random.truncated_normal([2], stddev=0.1)) # 一行三列,方差为0.1


a = 0.1  # 学习率为0.1
epoch = 500  # 循环500轮
# 训练部分
for epoch in range(epoch):  # 数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-y*)^2)
        # 计算loss对w, b的梯度
        grads = tape.gradient(loss, [w1, b1])
        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(a * grads[0])  # 参数w1自更新
        b1.assign_sub(a * grads[1])  # 参数b自更新

# 测试部分
total_correct, total_number = 0, 0
for x_test, y_test in test_db:
    # 前向传播求概率
    y = tf.matmul(x_test, w1) + b1
    y = tf.nn.softmax(y)
    predict = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
    # 将predict转换为y_test的数据类型
    predict = tf.cast(predict, dtype=y_test.dtype)
    # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
    correct = tf.cast(tf.equal(predict, y_test), dtype=tf.int32)
    # 将每个batch的correct数加起来
    correct = tf.reduce_sum(correct)
    # 将所有batch中的correct数加起来
    total_correct += int(correct)
    # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
    total_number += x_test.shape[0]
# 总的准确率等于total_correct/total_number
acc = total_correct / total_number
print("测试准确率 = %.2f %%" % (acc * 100.0))
# my_test = np.array([[5.9, 3.0, 5.1, 1.8]])
# print("输入 5.9  3.0  5.1  1.8")
# my_test = tf.convert_to_tensor(my_test)
# my_test = tf.cast(my_test, tf.float32)
# y = tf.matmul(my_test, w1) + b1
# y = tf.nn.softmax(y)
# species = {0: "狗尾鸢尾", 1: "杂色鸢尾", 2: "弗吉尼亚鸢尾"}
# predict = np.array(tf.argmax(y, axis=1))[0]  # 返回y中最大值的索引，即预测的分类
# print("该鸢尾花为：" + species.get(predict))