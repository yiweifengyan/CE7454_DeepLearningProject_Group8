# Readin all the training images and transform them into tensors
# Created by Shaun on 2018-10-18
# version 0.0, By Shaun, realized reading in all images in a floader and transformation

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
        print("-----------")
        print(root)   #os.walk()所在目录
        return(files)   #os.walk()所在目录的所有非目录文件名

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
print(train_data[9])
    
