#coding=utf-8
import os
from SamplingProcess import SamplingProcess
from ConstValue.global_variable import DATA_PATH
for root,subdir,files in os.walk(DATA_PATH):
    for file in files:
        print(file +" is sampling")
        sample = SamplingProcess(os.path.join(root, file))
        sample.sampling(0.5,256)