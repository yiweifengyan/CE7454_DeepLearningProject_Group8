# Readin all the training images and transform them into tensors
# Created by Shaun on 2018-10-18
# version 0.0, By Shaun, realized reading in all images in a folder and transformation, 2018-10-19
# version 1.0, By Shaun, realized reading in all labels in a folder and trans them into a tensor, 2018-10-20
import csv
from PIL import Image
import os
import torchvision
import torch

imdir = os.getcwd()

# Obtain the file names in the floder
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return (files)

def read_data_with_rps(folders):
    data_root = os.getcwd() + '/' + folders[0]
    data_names = [z.split('.')[0] for z in file_name(data_root)]
    # Read in the images and transform them into tensors
    data = torch.zeros([len(data_names), 3, 360, 640], dtype=torch.uint8)
    print(data.size())
    for i in range(len(data_names)):
        imgname = data_root + '/' + data_names[i] + '.jpg'
        im = Image.open(imgname)
        imtensor = torchvision.transforms.ToTensor()(im) * 255
        data[i] = imtensor

    # Define a function to convert labels into integters
    def getclass(name):
        if "boat1" == name:
            return 1
        if "boat2" == name:
            return 2
        if "boat3" == name:
            return 3
        if "boat5" == name:
            return 4
        if "car1" == name :
            return 5
        if "car12" == name:
            return 6
        if "car17" == name:
            return 7
        if "car20" == name:
            return 8
        if "person11" == name:
            return 9
        if "person19" == name:
            return 10
        if "person22" == name:
            return 11
        if "person29" == name:
            return 12
        if "riding3" == name:
            return 13
        if "riding8" == name:
            return 14
        if "truck1" == name:
            return 15
        print("error")
        return 0

    # For XML parse
    from xml.dom.minidom import parse
    import xml.dom.minidom

    # Get dictionary
    label_root = os.getcwd() + '/' + folders[1]

    # There are five integters in the label: class, xmin, xmax, yminx ymax
    # Thus it's a matrix [items num, 5]
    label = torch.zeros([len(data_names), 5], dtype=torch.int64)
    print(label.type())
    print(label.size())
    for i in range(len(data_names)):
        labname = label_root + '/' + data_names[i] + '.xml'
        lab = parse(labname)
        labcoll = lab.documentElement

        labclass = labcoll.getElementsByTagName("name")[0].childNodes[0]
        objname = labclass.nodeValue
        label[i][0] = getclass(objname)

        labxmin = labcoll.getElementsByTagName("xmin")[0].childNodes[0]
        xmin = int(labxmin.nodeValue)

        labxmax = labcoll.getElementsByTagName("xmax")[0].childNodes[0]
        xmax = int(labxmax.nodeValue)

        labymin = labcoll.getElementsByTagName("ymin")[0].childNodes[0]
        ymin = int(labymin.nodeValue)

        labymax = labcoll.getElementsByTagName("ymax")[0].childNodes[0]
        ymax = int(labymax.nodeValue)

        label[i][1] = xmin  # left top x
        label[i][2] = ymin  # left top y
        label[i][3] = xmax - xmin + 1  # width
        label[i][4] = ymax - ymin + 1  # height

    ## There are a specific number of region proposals for each image
    rp_root = os.getcwd() + '/' + folders[2]
    rps_4_all_images = []
    rp_labels_4_all_images = []
    for i in range(len(data_names)):
        rps = []
        rp_labels = []
        rpname = rp_root + '/' + data_names[i] + '.csv'
        with open(rpname, 'r') as infile:
            csvreader = csv.reader(infile)
            csvreader.__next__()
            for r, row in enumerate(csvreader):
                rps.append((int(row[0]), int(row[1]), int(row[2]), int(row[3])))
                rp_labels.append(int(row[4]))
        rps_4_all_images.append([z for z in rps])
        rp_labels_4_all_images.append([z for z in rp_labels])
    return data, label, rps_4_all_images, rp_labels_4_all_images


def read_data (folders):
    data_root = os.getcwd() + '/' + folders[0]
    data_names = [z.split('.')[0] for z in file_name(data_root)]
    # Read in the images and transform them into tensors
    data = torch.zeros([len(data_names), 3, 360, 640], dtype=torch.uint8)
    print(data.size())
    for i in range(len(data_names)):
        imgname = data_root + '/' + data_names[i] + '.jpg'
        im = Image.open(imgname)
        imtensor = torchvision.transforms.ToTensor()(im) * 255
        data[i] = imtensor

    # Define a function to convert labels into integters
    def getclass(name):
        if "boat1" == name:
            return 1
        if "boat2" == name:
            return 2
        if "boat3" == name:
            return 3
        if "boat5" == name:
            return 4
        if "car1" == name :
            return 5
        if "car12" == name:
            return 6
        if "car17" == name:
            return 7
        if "car20" == name:
            return 8
        if "person11" == name:
            return 9
        if "person19" == name:
            return 10
        if "person22" == name:
            return 11
        if "person29" == name:
            return 12
        if "riding3" == name:
            return 13
        if "riding8" == name:
            return 14
        if "truck1" == name:
            return 15
        print("error")
        return 0

    # For XML parse
    from xml.dom.minidom import parse
    import xml.dom.minidom

    # Get dictionary
    label_root = os.getcwd() + '/' + folders[1]

    # There are five integters in the label: class, xmin, xmax, yminx ymax
    # Thus it's a matrix [items num, 5]
    label = torch.zeros([len(data_names), 5], dtype=torch.int64)
    print(label.type())
    print(label.size())
    for i in range(len(data_names)):
        labname = label_root + '/' + data_names[i] + '.xml'
        lab = parse(labname)
        labcoll = lab.documentElement

        labclass = labcoll.getElementsByTagName("name")[0].childNodes[0]
        objname = labclass.nodeValue
        label[i][0] = getclass(objname)

        labxmin = labcoll.getElementsByTagName("xmin")[0].childNodes[0]
        xmin = int(labxmin.nodeValue)

        labxmax = labcoll.getElementsByTagName("xmax")[0].childNodes[0]
        xmax = int(labxmax.nodeValue)

        labymin = labcoll.getElementsByTagName("ymin")[0].childNodes[0]
        ymin = int(labymin.nodeValue)

        labymax = labcoll.getElementsByTagName("ymax")[0].childNodes[0]
        ymax = int(labymax.nodeValue)

        label[i][1] = xmin  # left top x
        label[i][2] = ymin  # left top y
        label[i][3] = xmax - xmin + 1  # width
        label[i][4] = ymax - ymin + 1  # height

    return data, label

