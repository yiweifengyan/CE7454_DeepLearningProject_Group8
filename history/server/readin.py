# Readin all the training images and transform them into tensors
# Created by Shaun on 2018-10-18
# version 0.0, By Shaun, realized reading in all images in a folder and transformation, 2018-10-19
# version 1.0, By Shaun, realized reading in all labels in a folder and trans them into a tensor, 2018-10-20
# version 2.0, By Shaun, difined as functions and can read in any number of images and xmls, 2018-10-25
# version 3.0, By Shaun, to run on Ubuntu server and the xml has been trans into [name, xmin, ymin, x-width, y-hight]

from PIL import Image
import os
import torchvision
import torch

# Obtain the file names in the floder
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):  
        return(files) 
    
# Define a function to convert labels into integters 
def getclass(name):
    if "boat" in name :
        return 0
    if "building" in name:
        return 1
    if "car" in name :
        return 2
    if "drone" in name:
        return 3
    if "group" in name :
        return 4
    if "horseride" in name:
        return 5
    if "paraglider" in name :
        return 6
    if "person" in name:
        return 7
    if "riding" in name :
        return 8
    if "truck" in name:
        return 9
    if "wakeboard" in name :
        return 10
    if "whale" in name:
        return 11
    return 12
    
    
def read_image(folder):
    train_data_root = folder
    train_data_names=file_name(train_data_root)
    print(train_data_names[0])
    # Read in the images and transform them into tensors
    train_data=torch.zeros([len(train_data_names),3,360,640],dtype=torch.float)
    for i in range(len(train_data_names)):
        imgname=train_data_root+'/'+train_data_names[i]
        im=Image.open(imgname)
        imtensor=torchvision.transforms.ToTensor()(im)
        train_data[i]=imtensor
    train_data=train_data*255
    return(train_data)

# Read in part of the images and transform them into tensors
def read_part_image(folder,start,length):
    train_data_root = folder
    train_data_names=file_name(train_data_root)
    
    train_data=torch.zeros([length,3,360,640],dtype=torch.float)
    local_num=0
    for i in range(len(train_data_names)):
        if (i>=start) and (i<start+length):
            imgname=train_data_root+'/'+train_data_names[i]
            im=Image.open(imgname)
            imtensor=torchvision.transforms.ToTensor()(im)
            train_data[local_num]=imtensor
            local_num=local_num+1
    train_data=train_data*255
    return(train_data)

# For XML parse
from xml.dom.minidom import parse
import xml.dom.minidom

def read_xml(floder):
    # Get dictionary
    train_label_root = floder
    train_label_names=file_name(train_label_root)
    
    # There are five integters in the label: class, xmin, ymin,x-width, y-hight
    # Thus it's a matrix [items num, 5]
    train_label=torch.zeros([len(train_label_names),5], dtype=torch.int64)
    for i in range(len(train_label_names)):
        labname=train_label_root+'/'+train_label_names[i]
        lab=parse(labname)
        labcoll=lab.documentElement
    
        labclass=labcoll.getElementsByTagName("name")[0].childNodes[0]
        objname=labclass.nodeValue
        train_label[i][0]=getclass(objname)
    
        labxmin=labcoll.getElementsByTagName("xmin")[0].childNodes[0]
        xmin=labxmin.nodeValue
        train_label[i][1]=int(xmin)
    
        labxmax=labcoll.getElementsByTagName("xmax")[0].childNodes[0]
        xmax=labxmax.nodeValue
        train_label[i][3]=int(xmax)-int(xmin)
    
        labymin=labcoll.getElementsByTagName("ymin")[0].childNodes[0]
        ymin=labymin.nodeValue
        train_label[i][2]=int(ymin)
    
        labymax=labcoll.getElementsByTagName("ymax")[0].childNodes[0]
        ymax=labymax.nodeValue
        train_label[i][4]=int(ymax)-int(ymin)

    return(train_label)

def read_part_xml(floder,start,length):
    # Get dictionary
    train_label_root = floder
    train_label_names=file_name(train_label_root)
    
    # There are five integters in the label: class, xmin, ymin, x-width, y-hight
    # Thus it's a matrix [items num, 5]
    train_label=torch.zeros([length,5], dtype=torch.int64)
    local_num=0
    for i in range(len(train_label_names)):
        if (i>=start) and (i<start+length):
            labname=train_label_root+'/'+train_label_names[i]
            lab=parse(labname)
            labcoll=lab.documentElement
            
            labclass=labcoll.getElementsByTagName("name")[0].childNodes[0]
            objname=labclass.nodeValue
            train_label[local_num][0]=getclass(objname)
            
            labxmin=labcoll.getElementsByTagName("xmin")[0].childNodes[0]
            xmin=labxmin.nodeValue
            train_label[local_num][1]=int(xmin)
            
            labxmax=labcoll.getElementsByTagName("xmax")[0].childNodes[0]
            xmax=labxmax.nodeValue
            train_label[local_num][3]=int(xmax)-int(xmin)
            
            labymin=labcoll.getElementsByTagName("ymin")[0].childNodes[0]
            ymin=labymin.nodeValue
            train_label[local_num][2]=int(ymin)
            
            labymax=labcoll.getElementsByTagName("ymax")[0].childNodes[0]
            ymax=labymax.nodeValue
            train_label[local_num][4]=int(ymax)-int(ymin)
            
            local_num=local_num+1
            
    return(train_label)


# a function that has the same read_data

def read_data(folders):
    data_dir=folders[0]
    data=read_image(data_dir)
    label_dir=folders[1]
    label=read_xml(label_dir)
    return(data,label)
