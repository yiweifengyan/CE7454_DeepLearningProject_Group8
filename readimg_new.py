# Readin all the training images and transform them into tensors
# Created by Shaun on 2018-10-18
# version 0.0, By Shaun, realized reading in all images in a folder and transformation, 2018-10-19
# version 1.0, By Shaun, realized reading in all labels in a folder and trans them into a tensor, 2018-10-20

from PIL import Image
import os
import torchvision
import torch

imdir=os.getcwd()
#print(imdir)
#imgname=imdir+'\\train_data\\000001.jpg'
#print(imgname)
#im=Image.open(imgname)
#im.show()

# Obtain the file names in the floder
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        #print("-----------")
        #print(root)   
        return(files)

#print(train_data_names)

def read_data(folders):
    train_data_root = os.getcwd() + '\\' + folders[0]
    train_data_names = file_name(train_data_root)
    # Read in the images and transform them into tensors
    train_data=torch.zeros([len(train_data_names),3,360,640], dtype=torch.uint8)
    print(train_data.size())
    for i in range(len(train_data_names)):
        imgname=train_data_root+'\\'+train_data_names[i]
        im=Image.open(imgname)
        imtensor=torchvision.transforms.ToTensor()(im)*255
        train_data[i]=imtensor
    #print(train_data[9])


    # Define a function to convert labels into integters
    def getclass(name):
        if "boat" in name:
            return 1
        if "building" in name:
            return 2
        if "car" in name :
            return 3
        if "drone" in name:
            return 4
        if "group" in name :
            return 5
        if "horseride" in name:
            return 6
        if "paraglider" in name :
            return 7
        if "person" in name:
            return 8
        if "riding" in name :
            return 9
        if "truck" in name:
            return 10
        if "wakeboard" in name :
            return 11
        if "whale" in name:
            return 12
        return 0

    # For XML parse
    from xml.dom.minidom import parse
    import xml.dom.minidom

    # Get dictionary
    train_label_root = os.getcwd() + '\\' + folders[1]
    train_label_names=file_name(train_label_root)
    #print(train_label_names)

    # There are five integters in the label: class, xmin, xmax, yminx ymax
    # Thus it's a matrix [items num, 5]
    train_label=torch.zeros([len(train_label_names),5], dtype=torch.int64)
    print(train_label.type())
    print(train_label.size())
    for i in range(len(train_label_names)):
        labname=train_label_root+'\\'+train_label_names[i]
        lab=parse(labname)
        labcoll=lab.documentElement

        labclass=labcoll.getElementsByTagName("name")[0].childNodes[0]
        objname=labclass.nodeValue
        train_label[i][0]=getclass(objname)

        labxmin=labcoll.getElementsByTagName("xmin")[0].childNodes[0]
        xmin=int(labxmin.nodeValue)

        labxmax=labcoll.getElementsByTagName("xmax")[0].childNodes[0]
        xmax=int(labxmax.nodeValue)

        labymin=labcoll.getElementsByTagName("ymin")[0].childNodes[0]
        ymin=int(labymin.nodeValue)

        labymax=labcoll.getElementsByTagName("ymax")[0].childNodes[0]
        ymax=int(labymax.nodeValue)

        train_label[i][1] = xmin    # left top x
        train_label[i][2] = ymax    # left top y
        train_label[i][3] = xmax - xmin + 1    # width
        train_label[i][4] = ymax - ymin + 1   # height

    return train_data, train_label
    
#print(train_label[0])
