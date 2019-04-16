from preTrain import *
import tensorflow as tf
from PreTrain.creat2TFRecord import  *
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