import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from VGG16 import VGG16
def getSavePath(savePath, modelName, fold, epoch):
    epochSavePath = os.path.join(savePath,modelName, "fold_" + str(fold), "epoch_" + str(epoch) + "/")
    foldSavePath = os.path.join(savePath,modelName, "fold_" + str(fold) + "/")
    modelPath = os.path.join(savePath,modelName+"/")
    if not os.path.exists(modelPath):
        os.mkdir(modelPath)
    if not os.path.exists(foldSavePath):
        os.mkdir(foldSavePath)
    if not os.path.exists(epochSavePath):
        os.mkdir(epochSavePath)
    return modelPath,foldSavePath, epochSavePath

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

def ModelSaver(ModelSavePath,sess,saver,global_step):
    saver.save(sess, save_path=ModelSavePath, global_step = global_step)
def lossPainter(lossPicSavepath,loss,batch_counter):
    plt.plot(loss)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(lossPicSavepath +"batch_"+ str(batch_counter)+ "_loss.jpg")
    plt.close()
def creatModel(ModelName,x_imgs):
    # 创建模型
    model = ""
    if ModelName == "VGG16":
        model = VGG16(x_imgs)
    elif ModelName == "ResNet":
        pass
    elif ModelName == "InceptionNet":
        pass
    else:
        raise Exception("Error Input~")
    return model