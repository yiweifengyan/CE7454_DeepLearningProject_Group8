#This is the YOLOv2 version of the project
#The architecture of the neural network
#Version 0.0 by Shuan, 2018-10-11
#Finished first 18 layers of YOLOv2 by Shaun, 2018-10-12
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import time
import utils

class simplynet(nn.Module):
    
    def __init__(self):
        super(simplynet,self).__init__()
    
        # layer1, 3*640*360  --> 32*640*360 
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1,bias=False)
        # maxpool,32*640*360 --> 32*320*180
        self.pool1=nn.MaxPool2d(2,2)
    
        # layer2, 32*320*180 --> 64*320*180 
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        # maxpool,64*320*180 --> 64*160*90
        self.pool2=nn.MaxPool2d(2,2)
    
        # layer3, 64*160*90 --> 128*160*90 
        self.conv3=nn.Conv2d(64,128,kernel_size=1,padding=1)
    
        # layer4, 128*160*90 --> 64*160*90 
        self.conv4=nn.Conv2d(128,64,kernel_size=3,padding=1)
    
        # layer5, 64*160*90 --> 128*160*90 
        self.conv5=nn.Conv2d(64,128,kernel_size=3,padding=1)
        # maxpool,128*160*90 --> 128*80*45
        self.pool5=nn.MaxPool2d(2,2)
    
        # layer6, 128*80*45 --> 256*80*45 
        self.conv6=nn.Conv2d(128,256,kernel_size=3,padding=1)
        
        # layer7, 256*80*45 --> 128*80*45 
        self.conv7=nn.Conv2d(256,128,kernel_size=1,padding=1)
        
        # layer8, 128*80*45 --> 256*80*45 
        self.conv8=nn.Conv2d(128,256,kernel_size=3,padding=1)
        # maxpool,256*80*45 --> 256*40*22.5?
        self.pool8=nn.MaxPool2d(2,2)
        
        # layer9, 256*40*22 --> 512*40*22 
        self.conv9=nn.Conv2d(256,512,kernel_size=3,padding=1)
        
        # layer10, 512*40*22 --> 256*40*22
        self.conv10=nn.Conv2d(512,256,kernel_size=1,padding=1)
        
        # layer11, 256*40*22 --> 512*40*22 
        self.conv11=nn.Conv2d(256,512,kernel_size=3,padding=1)
        
        # layer12, 512*40*22 --> 256*40*22
        self.conv12=nn.Conv2d(512,256,kernel_size=1,padding=1)
        
        # layer13, 256*40*22 --> 512*40*22 
        self.conv13=nn.Conv2d(256,512,kernel_size=3,padding=1)
        # maxpool, 512*40*22 --> 512*20*11
        self.pool13=nn.MaxPool2d(2,2)
        
        # layer14, 512*20*11 --> 1024**20*11
        self.conv14=nn.Conv2d(512,1024,kernel_size=3,padding=1)
        
        # layer15, 1024*20*11 --> 512*20*11
        self.conv15=nn.Conv2d(1024,512,kernel_size=1,padding=1)
        
        # layer16, 512*20*11 --> 1024*20*11
        self.conv16=nn.Conv2d(512,1024,kernel_size=3,padding=1)
        
        # layer17, 1024*20*11 --> 512*20*11
        self.conv17=nn.Conv2d(1024,512,kernel_size=1,padding=1)
        
        # layer18, 512*20*11 --> 1024*20*11
        self.conv18=nn.Conv2d(512,1024,kernel_size=3,padding=1)
        
        
        
        
        
    def forward(self,x):
        
        # Layer1,activation=leaky
        x=self.conv1(x)
        x=F.leaky_relu(x)
        x=self.pool1(x)
        
        # layer2,activation=leaky
        x=self.conv2(x)
        x=F.leaky_relu(x)
        x=self.pool2(x)
        
        # layer3,activation=leaky
        x=self.conv3(x)
        x=F.leaky_relu(x)
        
        # layer4,activation=leaky
        x=self.conv4(x)
        x=F.leaky_relu(x)
        
        # layer5,activation=leaky
        x=self.conv5(x)
        x=F.leaky_relu(x)

        # layer6,activation=leaky
        x=self.conv2(x)
        x=F.leaky_relu(x)
        x=self.pool6(x)
        
        
        
        return(x)
