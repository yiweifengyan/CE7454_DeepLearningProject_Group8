# Readin all the training images and transform them into tensors
# Created by Shaun on 2018-10-18
# version 0.0, By Shaun, realized reading in all images in a folder and transformation, 2018-10-19
# version 1.0, By Shaun, realized reading in all labels in a folder and trans them into a tensor, 2018-10-20
# version 2.0, By Shaun, difined as functions and can read in any number of images and xmls, 2018-10-25
# version 5.0, By Shaun, difine 80 classes, 2018-11-1

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
    if "boat1" in name :
        return 0
    if "boat2" in name :
        return 0
    if "boat3" in name :
        return 0
    if "boat4" in name :
        return 1
    if "boat5" in name :
        return 2
    if "boat6" in name :
        return 3
    if "boat7" in name :
        return 4
    if "boat8" in name :
        return 5
    if "building1" in name:
        return 6
    if "building2" in name:
        return 7
    if "building3" in name:
        return 8
    if ("car1" in name) and (len(name)==4) :
        return 9
    if ("car2" in name) and (len(name)==4) :
        return 10
    if "car3" in name :
        return 11
    if "car4" in name :
        return 11
    if "car5" in name :
        return 11
    if "car6" in name :
        return 12
    if "car8" in name :
        return 12
    if "car9" in name :
        return 12
    if "car10" in name :
        return 13
    if "car11" in name :
        return 14
    if "car12" in name :
        return 15
    if "car13" in name :
        return 16
    if "car14" in name :
        return 17
    if "car15" in name :
        return 18
    if "car16" in name :
        return 19
    if "car17" in name :
        return 20
    if "car18" in name :
        return 21
    if "car19" in name :
        return 22
    if "car20" in name :
        return 23
    if "car21" in name :
        return 24
    if "car22" in name :
        return 25
    if "car23" in name :
        return 26
    if "car24" in name :
        return 27
    if "drone1" in name:
        return 28
    if "drone2" in name:
        return 29
    if "drone3" in name:
        return 30
    if "drone4" in name:
        return 28
    if "group2" in name :
        return 31
    if "group3" in name :
        return 32
    if "horseride" in name:
        return 33
    if "paraglider" in name :
        return 34
    if ("person1" in name)and(len(name)==7):
        return 35
    if ("person2" in name)and(len(name)==7):
        return 36
    if "person3" in name:
        return 37
    if "person4" in name:
        return 38
    if "person5" in name:
        return 39
    if "person6" in name:
        return 40
    if "person7" in name:
        return 41
    if "person8" in name:
        return 42
    if "person9" in name:
        return 43
    if "person10" in name:
        return 44
    if "person11" in name:
        return 44
    if "person12" in name:
        return 45
    if "person13" in name:
        return 45
    if "person14" in name:
        return 46
    if "person15" in name:
        return 47
    if "person16" in name:
        return 48
    if "person17" in name:
        return 49
    if "person18" in name:
        return 50
    if "person19" in name:
        return 50
    if "person20" in name:
        return 51
    if "person21" in name:
        return 52
    if "person22" in name:
        return 53
    if "person23" in name:
        return 54
    if "person24" in name:
        return 55
    if "person25" in name:
        return 56
    if "person26" in name:
        return 57
    if "person27" in name:
        return 58
    if "person28" in name:
        return 59
    if "person29" in name:
        return 60
    if ("riding1" in name) and (len(name)==7) :
        return 61
    if "riding2" in name :
        return 62
    if "riding3" in name :
        return 63
    if "riding4" in name :
        return 64
    if "riding5" in name :
        return 65
    if "riding6" in name :
        return 66
    if "riding7" in name :
        return 67
    if "riding8" in name :
        return 68
    if "riding9" in name :
        return 68
    if "riding10" in name :
        return 69
    if "riding11" in name :
        return 70
    if "riding12" in name :
        return 71
    if "riding13" in name :
        return 72
    if "riding14" in name :
        return 72
    if "riding15" in name :
        return 73
    if "riding16" in name :
        return 74
    if "riding17" in name :
        return 75
    if "truck1" in name:
        return 76
    if "truck2" in name:
        return 77
    if "wakeboard" in name :
        return 78
    if "whale" in name:
        return 79
    return 80
    

    
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
