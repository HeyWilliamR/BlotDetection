import os
import random
import xml
from ConstValue.global_variable import IMG_WIDTH,IMG_HEIGHT
import cv2
from PreTrain.labelParser import LabelParserHandler

class preTrainProcess:
    def __init__(self,file,mergeFlag = False):
        self.file = file
        self.annotationfile = self.__mapToAnnotation()
        self.mergeFlag = mergeFlag
        self.labelRegion = self.__getGroundTruthArear(mergeFlag)
        self.image = cv2.imread(self.file)
        self.sampleSavePath = self.__getSampleSavePath()

    # 采样存储路径
    def __getSampleSavePath(self):
        path = os.path.dirname(self.file)
        path = os.path.dirname(path)
        return os.path.join(path,"sample")

    # 根据文件名映射到标记注释
    def __mapToAnnotation(self):
        path = os.path.dirname(self.file)
        path =os.path.dirname(path)
        filename = self.file.split("\\")[-1]
        attionationFilename = filename.split(".")[0] + ".xml"
        return os.path.join(path,"annotation",attionationFilename);

    # #对图片进行采样获取训练的样本，由于目前nut数据太少不在进行nut数据采样
    def sampling(self,ratio,sampleNums):
        filename = self.file.split("\\")[-1].split(".")[0]
        #计算采样数
        posGoal = sampleNums * ratio
        negGoal = sampleNums - posGoal
        backgroundSampleNum = 0
        sampleCounter = {}
        for labelname in self.labelRegion:
            sampleCounter[labelname] = 0
        sampleFlag = True
        size = self.image.shape
        while(backgroundSampleNum < negGoal):
            storeFlag = True
            #本次采样随机变换标志
            TransferFlag = random.choice([True, False])
            y1 = random.randint(0, size[0]-IMG_HEIGHT)
            x1 = random.randint(0, size[1]-IMG_WIDTH)
            print()
            x2 = x1+IMG_WIDTH
            y2 = y1+IMG_HEIGHT
            coordinate = ([y1,x1],[y2,x2])
            # 获取采样标记
            sign,crossrate = self.__getSampleSign(coordinate)
            if "again" == sign:
                continue
            elif "background"==sign :
                backgroundSampleNum += 1
                storeIndex = backgroundSampleNum
                print("The background %d's corssRate is %f the cooridnate is [(%d,%d),(%d,%d)]"
                      %(storeIndex,crossrate,x1,y1,x2,y2))
            else:
                sampleCounter[sign] += 1
                storeIndex = sampleCounter[sign]
                if sampleCounter[sign] > posGoal:
                    storeFlag = False
            if storeFlag:
                storeFile = os.path.join(self.sampleSavePath,sign,filename+" "+str(storeIndex)+".jpg")
                sample = self.image[y1:y2,x1:x2]
                if TransferFlag:
                    # 进行随机变换
                    pass
                cv2.imwrite(storeFile,sample)
        # 进行正例样本采样
        for item in sampleCounter:
            while sampleCounter[item]<posGoal:
                index = random.randint(0,len(self.labelRegion[item])-1)
                sampleCooridnate = self.labelRegion[item][index]
                storeIndex = sampleCounter[item]
                transferFlag = random.choice([True,False])
                if transferFlag :
                    pass
                storeFile = os.path.join(self.sampleSavePath, item, filename + " " + str(storeIndex) + ".jpg")
                sample = self.__getOrientalPositionSample(sampleCooridnate)
                cv2.imwrite(storeFile, sample)
                sampleCounter[item] += 1


    # 在指定位置采样
    def __getOrientalPositionSample(self, cooridnate):
        maxx = cooridnate["xmax"]
        maxy = cooridnate["ymax"]
        minx = cooridnate["xmin"]
        miny = cooridnate["ymin"]
        try:
            startX = max(0,maxx - IMG_WIDTH)
            Starty = max(0, maxy - IMG_HEIGHT)
            x1 = random.randint(startX, minx)
            y1 = random.randint(Starty, miny)
        except Exception:
            raise Exception("Sampling Size setting Error")
        x2 = x1 + IMG_WIDTH
        y2 = y1 + IMG_HEIGHT
        sample = self.image[x1:x2, y1:y2]
        return sample

    # 获取标记位置处的图片
    def getTestImg(self):
        blotImgList = []
        nutImgList = []
        for item in self.blot:
            img = self.__getOrientalPositionSample(item)
            blotImgList.append(img)
        for item in self.nut:
            img = self.__getOrientalPositionSample(item)
            nutImgList.append(img)
        return blotImgList,nutImgList

    # #获取图片的标注信息

    def __getLabels(self):
        handler = LabelParserHandler()
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        parser.setContentHandler(handler)
        parser.parse(self.annotationfile)
        labels = handler.labelresult()
        # #去掉None元素
        for i in range(len(labels)):
            if labels[i]["name"] == "none" :
                del labels[i]
        return labels


    #  计算采样框与标记框的交集所占标记框的比率
    def __caculCrossRate(self, Reframe, GTframe):
        x1 = Reframe[0][0]
        y1 = Reframe[0][1]
        width1 = Reframe[1][0] - Reframe[0][0]
        height1 = Reframe[1][1] - Reframe[0][1]

        x2 = GTframe["xmin"]
        y2 = GTframe["ymin"]
        width2 = GTframe["xmax"] - GTframe["xmin"]
        height2 = GTframe["ymax"] - GTframe["ymin"]

        endx = max(x1 + width1, x2 + width2)
        startx = min(x1, x2)
        width = width1 + width2 - (endx - startx)

        endy = max(y1 + height1, y2 + height2)
        starty = min(y1, y2)
        height = height1 + height2 - (endy - starty)

        if width <= 0 or height <= 0:
            ratio = 0  # 重叠率为 0
        else:
            Area = width * height  # 两矩形相交面积
            Area2 = width2 * height2
            ratio = Area * 1. / Area2
        return ratio

    # #判别采样labels
    def __getSampleSign(self, coordinate):
        for categoryInfo in self.labelRegion:
            for item in self.labelRegion[categoryInfo]:
                # #计算交叠比率，若大于0.7则采样结果为螺栓 0.4~0.7重新采样 0.4以下为背景图片
                crossRate = self.__caculCrossRate(coordinate, item)
                if crossRate >= 0.7:
                    return categoryInfo,crossRate
                elif 0.4 <= crossRate < 0.7:
                    return "again",crossRate
        return "background",crossRate

    # 获取blots 和nuts 标签位置
    def __getGroundTruthArear(self,mergeFlag):
        labelArear = {}
        labels = self.__getLabels()
        for item in labels:
            elem = {}
            elem["xmin"] = eval(item["topleft"][1])
            elem["ymin"] = eval(item["topleft"][0])
            elem["xmax"] = eval(item["bottomright"][1])
            elem["ymax"] = eval(item["bottomright"][0])
            labelname = item["name"]
            if labelArear.__contains__(labelname):
                labelArear[labelname].append(elem)
            else:
                labelArear[labelname] = []
                labelArear[labelname].append(elem)
        return labelArear
    # 获取变换映射
    def __getSampleTransferMap(self):
        transferMap = {}
        transferMap["RandomNoise"] = False
        transferMap["GrayTransfer"] = False
        transferMap["Rote"] = False
        transferMap["affine"] = False
        transferMap["Mirror"] = False
        