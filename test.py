import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import os
import struct
import cv2

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class mnist_test():
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 下载并加载mnist数据
        self.model_path = "MNIST.model/model_cnn.ckpt"
        self.y_actual = tf.placeholder(tf.float32, shape=[None, 10])  # 输入的标签占位符
        self.x = tf.placeholder(tf.float32, [None, 784])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])  # 转换输入数据shape,以便于用于网络中

        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层
        h_pool1 = max_pool(h_conv1)  # 第一个池化层

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二个卷积层
        h_pool2 = max_pool(h_conv2)  # 第二个池化层

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层

        self.keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)  # dropout层

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        self.y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层 回归模型

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.model_path)  # 这里使用了之前保存的模型参数
        #self.prediction = tf.argmax(self.y_predict,1)
        self.prediction = self.y_predict

        self.cross_entropy = -tf.reduce_sum(self.y_actual * tf.log(self.y_predict))  # 交叉熵
        self.train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.cross_entropy)  # 梯度下降法
        self.correct_prediction = tf.equal(tf.argmax(self.y_predict, 1), tf.argmax(self.y_actual, 1))  # 预测结果与实际值对比
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))  # 精确度计算

    def classify(self,data):
        predint=self.prediction.eval(feed_dict={self.x: [data],self.keep_prob: 1.0}, session=self.sess)
        #predint = self.sess.run(tf.argmax(self.prediction, 1), feed_dict={self.x: data, self.keep_prob: 1.0})
        return predint[0]
    def train(self):
        for i in range(5000):
            # batch_x, batch_y = mnist.train.next_batch(50)
            batch = self.mnist.train.next_batch(50)
            self.train_step.run(feed_dict={self.x: batch[0], self.y_actual: batch[1], self.keep_prob: 0.5})
            if i % 100 == 0:  # 训练100次，验证一次
                # train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
                train_acc = self.sess.run(self.accuracy,
                                     feed_dict={self.x: batch[0], self.y_actual: batch[1], self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_acc))
        test_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images, self.y_actual: self.mnist.test.labels,
                                                 self.keep_prob: 1.0})
        # test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
        print("test accuracy %g" % test_acc)
        # Save model weights to disk
        save_path = self.saver.save(self.sess, self.model_path)


#test1 = mnist_test()
#test1.train()