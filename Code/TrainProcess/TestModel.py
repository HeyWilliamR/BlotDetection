import cv2
from Model.VGG16 import VGG16
from PreTrain.creat2TFRecord import *

def getAccuracy(sess,model,test_img_list,test_label_list):
    probs = model.probs
    correctCounter = 0
    maps = {"background":{},"blot":{},"nut":{},"blotwithnut":{}}
    for item1 in maps:
        for item2 in maps:
            maps[item1][item2] = 0
    for index in range(len(test_img_list)):
        img = cv2.imread(test_img_list[index])
        img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
        test_result = sess.run(probs,feed_dict={model.imgs:[img]})
        test_max_index = np.argmax(test_result)
        reallabel = test_label_list[index]
        if test_max_index == reallabel:
            correctCounter += 1
        if reallabel == 0:
            s = maps["background"]
        elif reallabel == 1:
            s = maps["blot"]
        elif reallabel == 2:
            s = maps["nut"]
        elif reallabel == 3:
            s = maps["blotwithnut"]
        if test_max_index == 0:
            s["background"] += 1
        elif test_max_index == 1:
            s["blot"] += 1
        elif test_max_index == 2:
            s["nut"] += 1
        elif test_max_index == 3:
            s["blotwithnut"] += 1
    print(len(test_label_list))
    return correctCounter/len(test_label_list),maps
if __name__ == "__main__":

    data = np.load("D:/Code/Python/BlotDetection/ModelSave/vgg16_weights.npz")
    x_imgs = tf.placeholder(dtype=tf.float32,shape = [None,IMG_WIDTH,IMG_HEIGHT,3])
    model = VGG16(x_imgs)

    saver = model.saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    modelsaved = tf.train.latest_checkpoint("D:/Code/Python/BlotDetection/ModelSave/20190415/fold_0/epoch_0/")
    saver.restore(sess,modelsaved)

    img_list,label_list = get_file(SAMPLE_PATH)
    accuracy = getAccuracy(sess,model,img_list,label_list)
    print(accuracy)