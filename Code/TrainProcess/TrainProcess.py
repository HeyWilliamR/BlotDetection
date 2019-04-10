
#coding=utf-8
import sys
sys.path.append("/home/xhsu/llg/BlotDetection/Code/")
sys.path.append("/home/xhsu/llg/BlotDetection/Code/PreTrain/")
sys.path.append("/home/xhsu/llg/BlotDetection/Code/Model/")
sys.path.append("/home/xhsu/llg/BlotDetection/Code/ConstValue/")
from creat2TFRecord import *
from global_variable import *
import tensorflow as tf
from VGG16 import VGG16
import VGG16 as model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from PreTrain.creat2TFRecord import *
import cv2




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
def trainProcess(modelName):
    x_imgs = tf.placeholder(tf.float32,shape = [None,IMG_WIDTH,IMG_HEIGHT,3])
    y_labels = tf.placeholder(tf.int32,shape = [None,2])
    batch_total = caculTotalBatch(SAMPLE_PATH,batchnum=BATCH_SIZE)
    imagelist, labellist = get_file(SAMPLE_PATH)
    for fold in range(FOLD_VALUE):
        print(" %d Flod  begin training"%(fold+1))
        model = creatModel(modelName, x_imgs)
        result = model.probs
        saver = model.saver()
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=y_labels))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
        x_trainSet, y_trainSet, x_testSet, y_testSet = split_dataset(imagelist,labellist,fold)
        image_batch, label_batch = get_batch(x_trainSet,y_trainSet,IMG_WIDTH,IMG_HEIGHT,batch_size=BATCH_SIZE,capcity=32)
        loss_list = []
        acc_list = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            try:
                for epoch in range(EPOCH_NUM):  # 每一轮迭代
                    epoch_loss_list = []
                    modelSavePath, flodModelSavepath, epochModelSavepath = getSavePath(MODEL_SAVE_PATH,modelName=modelName, fold=fold, epoch=epoch)
                    loss_file = open(os.path.join(epochModelSavepath,"losslog_" + str(epoch) + ".log"), "w")
                    print("epoch %d begin training"%(epoch+1))
                    for batchCounter in range(batch_total):
                            starttime = time.time()
                            # 获取每一个batch中batch_size个样本和标签
                            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
                            labels = one_hot(label_batch_v)
                            sess.run(optimizer, feed_dict={x_imgs: image_batch_v, y_labels: labels})
                            loss_record = sess.run(loss, feed_dict={x_imgs: image_batch_v, y_labels: labels})
                            endtime = time.time()
                            print("batch %d cost time : %d ms loss is %f"%(batchCounter+1,endtime-starttime,loss_record))
                            currentTimeStamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                            loss_file.write("%s the loss is %f\n" %(currentTimeStamp, loss_record))
                            loss_list.append(loss_record)
                            epoch_loss_list.append(loss_record)
                            if batchCounter != 0 and batchCounter % 50 == 0:
                                ModelSaver(ModelSavePath = epochModelSavepath,sess = sess,saver = saver,global_step = batchCounter)
                                lossPainter(lossPicSavepath = epochModelSavepath,loss=epoch_loss_list,batch_counter=batchCounter)
                    print("epoch %d end training" % (epoch+1))
                    ModelSaver(ModelSavePath=epochModelSavepath, sess=sess, saver=saver, global_step=batchCounter)
                    lossPainter(lossPicSavepath=epochModelSavepath, loss=epoch_loss_list,batch_counter = batchCounter)
                    acc = getAccuracy(sess, model=model, test_img_list=x_testSet, test_label_list=y_testSet)
                    acc_list.append(acc)
                    loss_file.write("now the epoch %d's accuracy is %f\n" % (epoch, acc))
                    print("now the epoch %d's accuracy is %f\n" % (epoch, acc))
                    loss_file.close()
            except tf.errors.OutOfRangeError:
                print("done")
            finally:
                coord.request_stop()
                ModelSaver(ModelSavePath=epochModelSavepath, sess=sess, saver=saver, global_step=batchCounter)
                lossPainter(lossPicSavepath=epochModelSavepath, loss=epoch_loss_list,batch_counter=batchCounter)
            coord.join(threads)
        lossPainter(lossPicSavepath=flodModelSavepath,loss= loss_list)
    accfile = open(os.path.join(modelSavePath,"accuracy.txt"),"w")
    accavg = np.average(acc_list)
    accfile.write("average accuracy is %f "%accavg)
    accfile.close()
if __name__ =="__main__":
    trainProcess("VGG16")