# select u% pictures from each class
# seperate the selected pictures into training set (train_u%) and testing set (test_u%)

import random as rm
from datetime import datetime
import os
from shutil import copyfile, rmtree

u = 0.1
train_u = 0.8
test_u = 0.2

raw_dir = os.getcwd() + "\\raw\\"
train_data_dir = os.getcwd() + "\\train_data\\"
train_label_dir = os.getcwd() + "\\train_label\\"
test_data_dir = os.getcwd() + "\\test_data\\"
test_label_dir = os.getcwd() + "\\test_label\\"

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
    for filename in os.listdir(raw_dir + cls + "/"):
        if ".jpg" not in filename:
            continue

        if rm.random() > u:
            continue

        filename_xml = filename.split('.')[0] + ".xml"

        if rm.randint(1, 100) > test_u * 100:   # train data
            copyfile(raw_dir + cls + "/" + filename, train_data_dir + cls + '_' + filename)
            copyfile(raw_dir + cls + "/" + filename_xml, train_label_dir + cls + '_' + filename_xml)
        else:                       # test data
            copyfile(raw_dir + cls + "/" + filename, test_data_dir + cls + '_' + filename)
            copyfile(raw_dir + cls + "/" + filename_xml, test_label_dir + cls + '_' + filename_xml)
