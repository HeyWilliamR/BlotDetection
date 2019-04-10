#coding=utf-8
from datetime import datetime
import math
import time
import tensorflow as tf
from ConstValue.global_variable import  *


class VGG16 :
    def __init__(self,imgs):
        self.imgs = imgs
        self.keep_prob = DROPOUT_RATE
        self.parameters = []
        self.convlayers()
        self.fc_layers()
        self.probs = self.fc8
    def batch_norm(self,inputs,is_training = True,is_conv_out= True, decay = 0.999):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]),trainable=False)
        if is_training:
            if is_conv_out:
                batch_mean,batch_var = tf.nn.moments(inputs,[0,1,2])
            else:
                batch_mean,batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean,pop_mean*decay+batch_mean*(1-decay))
            train_var = tf.assign(pop_var,pop_var *decay+batch_var*(1-decay))
            with tf.control_dependencies([train_mean,train_var]):
                return tf.nn.batch_normalization(inputs,batch_mean,batch_var,beta,scale,0.001)
        else:
            return tf.nn.batch_normalization(inputs,pop_mean,pop_var,beta,scale,0.001)
    def conv_op(self,input_op,name,kh,kw,n_out,dh,dw):
        n_in = input_op.get_shape()[-1]
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope+"w",shape = [kh,kw,n_in,n_out],dtype = tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding="SAME")
            bias_init_val = tf.constant(0.0,shape=[n_out],dtype=tf.float32)
            biases = tf.Variable(bias_init_val,trainable=True,name = "b")
            z = tf.nn.bias_add(conv,biases)
            activation = tf.nn.relu(z,name=scope)
            self.parameters += [kernel,biases]
            return activation
    def fc_op(self, input_op, name, n_out):
        shape = input_op.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] *shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_op,[-1,size])
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope+"w",shape=[size,n_out],dtype = tf.float32,
                                     initializer= tf.contrib.layers.xavier_initializer())
            biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name = "b")
            activateion = tf.nn.relu_layer(input_data_flat,kernel,biases,name = scope)
            self.parameters += [kernel,biases]
            return activateion
    def mpool_op(self, input_op, name ,kh, kw, dh, dw):
        return tf.nn.max_pool(input_op,ksize=[1, kh, kw, 1],strides=[1, dh, dw, 1],padding= "SAME",name = name)
    def convlayers(self):
        #conv1
        self.conv1_1 = self.conv_op(input_op=self.imgs,name="conv1_1"
                                    ,kh = 3, kw = 3,n_out = 64,dh = 1, dw = 1)
        self.conv1_2 = self.conv_op(input_op=self.conv1_1,name="conv1_2"
                                    ,kh = 3,kw =3,n_out = 64,dh =1 ,dw = 1)
        self.pool1 = self.mpool_op(self.conv1_2,name = "maxpool1",kh = 2, kw = 2, dw = 2, dh = 2)

        #conv2
        self.conv2_1 = self.conv_op(input_op = self.pool1,name = "conv2_1",kh =3, kw =3,
                                    n_out = 128, dh=1, dw = 1)
        self.conv2_2 = self.conv_op(input_op = self.conv2_1,name = "conv2_2",kh =3, kw = 3,
                                    n_out=128, dh=1, dw = 1)
        self.pool2 = self.mpool_op(self.conv2_2,name = "maxpool2", kh =2 ,kw = 2, dw = 2, dh = 2)

        #conv3
        self.conv3_1 = self.conv_op(input_op = self.pool2,name = "conv3_1", kh=3,kw=3,
                                    n_out = 256, dh = 1, dw =1)
        self.conv3_2 = self.conv_op(input_op = self.conv3_1,name = "conv3_2", kh=3,kw=3,
                                    n_out = 256, dh = 1,dw =1)
        self.conv3_3 = self.conv_op(input_op = self.conv3_2,name = "conv3_3", kh = 3, kw =3,
                                    n_out = 256,dh = 1,dw =1)
        self.pool3 = self.mpool_op(input_op=self.conv3_3,name="maxpool3",kh=2,kw=2,dw=2,dh=2)

        #conv4
        self.conv4_1 = self.conv_op(input_op=self.pool3,name = "conv4_1",kh = 3,kw = 3,
                                    n_out = 512,dh =1,dw =1)
        self.conv4_2 = self.conv_op(input_op=self.conv4_1,name="conv4_2",kh =3,kw=3,
                                    n_out=512,dh=1,dw=1)
        self.conv4_3 = self.conv_op(input_op = self.conv4_2,name = "conv4_3",kh = 3,kw =3 ,
                                    n_out = 512,dh = 1,dw = 1)
        self.pool4 = self.mpool_op(input_op=self.conv4_3,name="maxpool4",kh=2,kw=2,dh=2,dw=2)

        #conv5
        self.conv5_1 = self.conv_op(input_op=self.pool4,name = "conv5_1",kh=3,kw=3
                                    ,n_out=512,dh=1,dw=1)
        self.conv5_2 = self.conv_op(input_op=self.conv5_1,name = "conv5_2",kh=3,kw=3
                                    ,n_out=512,dh=1,dw=1)
        self.conv5_3 = self.conv_op(input_op = self.conv5_2,name = "conv5_3",kh = 3,kw =3
                                    ,n_out=512,dh = 1,dw = 1)
        self.pool5 = self.mpool_op(input_op=self.conv5_3,name = "maxpool5",kh=2,kw=2,dh=2,dw=2)
    def fc_layers(self):
        self.fc6 = self.fc_op(input_op=self.pool5,name = 'fc6',n_out=4096)
        self.fc6_drop = tf.nn.dropout(self.fc6,keep_prob=self.keep_prob,name = "fc6_drop")
        self.fc7 = self.fc_op(input_op=self.fc6_drop,name = "fc7",n_out = 4096)
        self.fc7_drop = tf.nn.dropout(self.fc7,keep_prob=self.keep_prob,name = "fc7_drop")
        self.fc8 = self.fc_op(input_op=self.fc7_drop,name="fc8",n_out = 2)
    def saver(self):
        return tf.train.Saver()
