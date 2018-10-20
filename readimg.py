# Readin all the training images and transform them into tensors
# Created by Shaun on 2018-10-18
# version 0.0, By Shaun, realized reading in all images in a folder and transformation, 2018-10-19
# version 1.0, By Shaun, realized reading in all labels in a folder and trans them into a tensor, 2018-10-20

from PIL import Image
import os
import torchvision


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

train_data_root = imdir+'\\train_data'
train_data_names=file_name(train_data_root)
#print(train_data_names)

# Read in the images and transform them into tensors
train_data=torch.zeros([len(train_data_names),3,360,640],dtype=torch.float64)
print(train_data.size())
for i in range(len(train_data_names)):
    imgname=train_data_root+'\\'+train_data_names[i]
    im=Image.open(imgname)
    imtensor=torchvision.transforms.ToTensor()(im)
    train_data[i]=imtensor
train_data=train_data*255
#print(train_data[9])
    
    
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

# For XML parse
from xml.dom.minidom import parse
import xml.dom.minidom

# Get dictionary
train_label_root = imdir+'\\train_label'
train_label_names=file_name(train_label_root)
#print(train_label_names)

# There are five integters in the label: class, xmin, xmax, yminx ymax
# Thus it's a matrix [items num, 5]
train_label=torch.zeros([len(train_label_names),5], dtype=torch.int32)
print(train_label.type())
print(train_label.size())
for i in range(len(train_label_names)):
    labname=train_label_root+'\\'+train_label_names[i]
    lab=parse(labname)
    labcoll=lab.documentElement
    
    labclass=labcoll.getElementsByTagName("name")[0].childNodes[0]
    objname=labclass.nodeValue
    #print(objname)
    #print(getclass(objname))
    train_label[i][0]=getclass(objname)
    
    labxmin=labcoll.getElementsByTagName("xmin")[0].childNodes[0]
    xmin=labxmin.nodeValue
    train_label[i][1]=int(xmin)
    #print(xmin)
    
    labxmax=labcoll.getElementsByTagName("xmax")[0].childNodes[0]
    xmax=labxmax.nodeValue
    train_label[i][2]=int(xmax)
    #print(xmax)
    
    labymin=labcoll.getElementsByTagName("ymin")[0].childNodes[0]
    ymin=labymin.nodeValue
    train_label[i][3]=int(ymin)
    #print(ymin)
    
    labymax=labcoll.getElementsByTagName("ymax")[0].childNodes[0]
    ymax=labymax.nodeValue
    train_label[i][4]=int(ymax)
    #print(ymax)
    
#print(train_label[0])
    
