from creat2TFRecord import *
from global_variable import *
import tensorflow as tf

import VGG16 as model
import matplotlib.pyplot as plt
import time
import cv2

def getSaveModelPath(savepath, fold):
    modelSavePath = os.path.join(savepath,"fold"+str(fold))
    if not os.path.exists(modelSavePath):
        os.mkdir(modelSavePath)
    return modelSavePath

def getAccuracy(sess,model,test_img_list,test_label_list):
    probs = model.probs
    correctCounter = 0
    for index in range(len(test_img_list)):
        img = cv2.imread(test_img_list[index])
        img = cv2.resize(img,(224,224))
        test_result = sess.run(probs,feed_dict={model.imgs:[img]})
        test_max_index = np.argmax(test_result)
        if test_max_index == test_label_list[index]:
            correctCounter += 1
    return correctCounter/len(test_label_list)



def trainProcess(Model,imglist,labellist):
    x_img = tf.placeholder(tf.float32,shape = [None,IMG_WIDTH,IMG_HEIGHT])