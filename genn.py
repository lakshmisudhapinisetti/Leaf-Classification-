

import os
import random
import shutil
from shutil import copy2

datadir_normal = "C:/Users/sudha/OneDrive/Desktop/this/dataset_origin"
folderlist = os.listdir(datadir_normal)
# datadir_normal = "./Annotations"
for folder in folderlist:
    inner_path = os.path.join(datadir_normal, folder)
    all_data = os.listdir(inner_path)
    num_all_data = len(all_data)
    print("num_all_data: " + str(num_all_data))
    index_list = list(range(num_all_data))
    # print(index_list)
    random.seed(1)
    random.shuffle(index_list)
    num = 0


    trainDir = "./train"  # （Put the training set in this folder）
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)


    validDir = "./val"  # （put the validation set in this folder）
    if not os.path.exists(validDir):
        os.mkdir(validDir)


    testDir = './test'  # （put the testing set in this folder）
    if not os.path.exists(testDir):
        os.mkdir(testDir)

    for i in index_list:
        fileName = os.path.join(inner_path, all_data[i])
        if num < num_all_data * 0.8:
            # print(str(fileName))
            copy2(fileName, trainDir)
        elif num > num_all_data * 0.8 and num <= num_all_data * 0.9:
            # print(str(fileName))
            copy2(fileName, validDir)
        else:
            copy2(fileName, testDir)
        num += 1

