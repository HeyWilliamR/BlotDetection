from SamplingTrainDataSet import SamplingTrainDataSet
from labelParser import labelParser
import os
import tensorflow as tf
import VGG16 as model
import time
import matplotlib.pyplot as plt
import creat_and_read_TFRecord2 as reader2
if __name__ == "__main__":
    annotation = "D:\\DataSet\\picture\\annotation"
    samplepath = "D:\\DataSet\\picture\\sample"
    rootpath = "D:\\DataSet\\picture\\"
    save_path = 'D:\\Code\\Python\\ObjectDetection\\model_save\\'
    parser = labelParser()
    sess = tf.Session()
    x_imgs = tf.placeholder(tf.float32,[None,224,224,3])
    y_imgs = tf.placeholder(tf.int32,[None,3])
    # dropout 调整
    vgg = model.vgg16(x_imgs, 0.4)
    model = tf.train.latest_checkpoint(save_path)
    saver = vgg.saver()
    saver.restore(sess,save_path=model)
    imglist = []
    # for item in os.listdir(rootpath):
    #     if os.path.isfile(item):
    #         imglist.append(item.split(".")[0])
    # for item in imglist:
    #     labelfile = os.path.join(annotation,item+".xml")
    #     image = os.path.join(rootpath,item+".jpg")
    #     labels = parser.getLabels(labelfile)
    #     dataSampling = TrainDataSet(labels,image,samplepath)
    #     dataSampling.Sampling(0.5,512)
    X_train,y_train = reader2.get_file(samplepath)
    image_batch,label_batch = reader2.get_batch(X_train,y_train,32,1024)

    result = vgg.probs
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result,labels = y_imgs))
    #调整学习率
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)
    count = -1
    totalloss = []
    try:
        while not coord.should_stop():
            batchLoss = []
            epochloss = []
            count += 1
            image, labels = sess.run([image_batch, label_batch])
            labels = reader2.one_hot(labels)
            start_time = time.time()
            for epoch in range(512):
                sess.run(optimizer,feed_dict={x_imgs:image,y_imgs:labels})
                loss_record = sess.run(loss,feed_dict={x_imgs:image,y_imgs:labels})
                print("now the loss is %f"%loss_record)
                batchLoss.append(loss_record)
                epochloss.append(loss_record)
                totalloss.append(loss_record)
                end_time = time.time()
                print("time: {}".format(end_time-start_time))
                start_time = end_time
                print("------------epoch %d is finished-----------------------------------" % epoch)
                if epoch % 50 ==0:
                    if epoch !=0:
                        saver.save(sess,save_path=save_path,global_step=epoch)
                if epoch % 10 == 0:
                    plt.plot(batchLoss)
                    plt.xlabel("iter")
                    plt.ylabel("loss")
                    plt.tight_layout()
                    plt.savefig("./losspicture/batch" + str(count))
                    plt.close()
                if epoch % 20 == 0:
                    if epoch == 0:
                        preminloss = loss_record
                    else:
                        currentminloss = min(epochloss)
                        delta = preminloss - currentminloss
                        print("delta bewteen pre 10 epoch and current 10 epoch is {}".format(delta))
                        if delta < 0.0001:
                            break;
                        else:
                            preminloss = currentminloss
                            epochloss.clear()
    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        saver.save(sess, save_path=save_path)
        print("optimization Finshed!")
        coord.request_stop()
        coord.join(threads)
        plt.plot(batchLoss)
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig("./losspicture/batchloss" + str(count))
