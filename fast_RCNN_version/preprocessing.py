# select u% pictures from each class
# seperate the selected pictures into training set (train_u%) and testing set (test_u%)

import random as rm
from datetime import datetime
import os
from shutil import copyfile, rmtree


train_u = 50
test_u = 50

raw_dir = "D:" + "\\raw\\"
train_data_dir = os.getcwd() + "\\segment-random\\train_data\\"
train_label_dir = os.getcwd() + "\\segment-random\\train_label\\"
test_data_dir = os.getcwd() + "\\segment-random\\test_data\\"
test_label_dir = os.getcwd() + "\\segment-random\\test_label\\"

# delete origin files in these folders
rmtree(train_data_dir)
os.makedirs(train_data_dir)
rmtree(train_label_dir)
os.makedirs(train_label_dir)
rmtree(test_data_dir)
os.makedirs(test_data_dir)
rmtree(test_label_dir)
os.makedirs(test_label_dir)

rm.seed(datetime.now())


for cls in os.listdir(raw_dir):
    if cls not in ["boat3", "boat5", "boat2", "boat1", "car1", "car17", "car12", "car20", "person19", "person22", "person11", "person29", "truck1", "riding8", "riding3"]:
        continue
    filenames = [z for z in os.listdir(raw_dir + cls + "/") if ".jpg" in z]
    file_num = len(filenames)

    selected = []
    idx = -1
    for i in range(train_u):
        while idx in selected or idx == -1:
            idx = rm.randint(0, file_num - 1)
        selected.append(idx)
        filename = filenames[idx]
        filename_xml = filename.split('.')[0] + ".xml"
        copyfile(raw_dir + cls + "/" + filename, train_data_dir + cls + '_' + filename)
        copyfile(raw_dir + cls + "/" + filename_xml, train_label_dir + cls + '_' + filename_xml)

    selected = []
    idx = -1
    for j in range(test_u):
        while idx in selected or idx == -1:
            idx = rm.randint(0, file_num - 1)
        selected.append(idx)
        filename = filenames[idx]
        filename_xml = filename.split('.')[0] + ".xml"
        copyfile(raw_dir + cls + "/" + filename, test_data_dir + cls + '_' + filename)
        copyfile(raw_dir + cls + "/" + filename_xml, test_label_dir + cls + '_' + filename_xml)
