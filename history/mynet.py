import torch
import torch.nn as nn
import torch.nn.functional as F

class simplynet(nn.Module):
    
    def __init__(self):
        super(simplynet,self).__init__()
    
        # layer1, 3*640*360  --> 32*640*360 
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        # maxpool,32*640*360 --> 32*320*180
        self.pool1=nn.MaxPool2d(2,2)
    
        # layer2, 32*320*180 --> 64*320*180 
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        # maxpool,64*320*180 --> 64*160*90
        self.pool2=nn.MaxPool2d(2,2)
    
        # layer3, 64*160*90 --> 128*160*90 
        self.conv3=nn.Conv2d(64,128,kernel_size=1,padding=0)
    
        # layer4, 128*160*90 --> 64*160*90 
        self.conv4=nn.Conv2d(128,64,kernel_size=3,padding=1)
    
        # layer5, 64*160*90 --> 128*160*90 
        self.conv5=nn.Conv2d(64,128,kernel_size=3,padding=1)
        # maxpool,128*160*90 --> 128*80*45
        self.pool5=nn.MaxPool2d(2,2)
    
        # layer6, 128*80*45 --> 256*80*45 
        self.conv6=nn.Conv2d(128,256,kernel_size=3,padding=1)
        
        # layer7, 256*80*45 --> 128*80*45 
        self.conv7=nn.Conv2d(256,128,kernel_size=1,padding=0)
        
        # layer8, 128*80*45 --> 256*80*45 
        self.conv8=nn.Conv2d(128,256,kernel_size=3,padding=1)
        # maxpool,256*80*45 --> 256*40*22.5?
        self.pool8=nn.MaxPool2d(2,2)
        
        # layer9, 256*40*22 --> 512*40*22 
        self.conv9=nn.Conv2d(256,512,kernel_size=3,padding=1)
        
        # layer10, 512*40*22 --> 256*40*22
        self.conv10=nn.Conv2d(512,256,kernel_size=1,padding=0)
        
        # layer11, 256*40*22 --> 512*40*22 
        self.conv11=nn.Conv2d(256,512,kernel_size=3,padding=1)
        
        # layer12, 512*40*22 --> 256*40*22
        self.conv12=nn.Conv2d(512,256,kernel_size=1,padding=0)
        
        # layer13, 256*40*22 --> 512*40*22 
        self.conv13=nn.Conv2d(256,512,kernel_size=3,padding=1)
        # maxpool, 512*40*22 --> 512*20*11
        self.pool13=nn.MaxPool2d(2,2)
        
        # layer14, 512*20*11 --> 1024**20*11
        self.conv14=nn.Conv2d(512,1024,kernel_size=3,padding=1)
        
        # layer15, 1024*20*11 --> 512*20*11
        self.conv15=nn.Conv2d(1024,512,kernel_size=1,padding=0)
        
        # layer16, 512*20*11 --> 1024*20*11
        self.conv16=nn.Conv2d(512,1024,kernel_size=3,padding=1)
        
        # layer17, 1024*20*11 --> 512*20*11
        self.conv17=nn.Conv2d(1024,512,kernel_size=1,padding=0)
        
        # layer18, 512*20*11 --> 1024
        self.conv18=nn.Conv2d(512,1024,kernel_size=3,padding=1)
        self.pool18=nn.MaxPool2d((11,20),stride=1)
        
        # layer19, 1024 --> 12
        self.line19=nn.Linear(1024,12,bias=True)
        
        
        
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
        x=self.pool5(x)

        # layer6,activation=leaky
        x=self.conv6(x)
        x=F.leaky_relu(x)
        
        # layer7,activation=leaky
        x=self.conv7(x)
        x=F.leaky_relu(x)
        
        # layer8,activation=leaky
        x=self.conv8(x)
        x=F.leaky_relu(x)
        x=self.pool8(x)
        
        # layer9,activation=leaky
        x=self.conv9(x)
        x=F.leaky_relu(x)
        
        # layer10,activation=leaky
        x=self.conv10(x)
        x=F.leaky_relu(x)
        
        # layer11,activation=leaky
        x=self.conv11(x)
        x=F.leaky_relu(x)
        
        # layer12,activation=leaky
        x=self.conv12(x)
        x=F.leaky_relu(x)
        
        # layer13,activation=leaky
        x=self.conv13(x)
        x=F.leaky_relu(x)
        x=self.pool13(x)
        
        # layer14,activation=leaky
        x=self.conv14(x)
        x=F.leaky_relu(x)
        
        # layer15,activation=leaky
        x=self.conv15(x)
        x=F.leaky_relu(x)
        
        # layer16,activation=leaky
        x=self.conv16(x)
        x=F.leaky_relu(x)
        
        # layer17,activation=leaky
        x=self.conv17(x)
        x=F.leaky_relu(x)
        
        # layer18,activation=leaky
        x=self.conv18(x)
        x=F.leaky_relu(x)
        x=self.pool18(x)
        x=x.view(-1,1024)
        
        # layer19
        x=self.line19(x)
        #x=x.view(-1,12)
        x=F.softmax(x,dim=1)
        
        
        return(x)
    
class simplynet_v2(nn.Module):
    
    def __init__(self):
        super(simplynet_v2,self).__init__()
    
        # layer1, 3*360*640  --> 32*360*640 
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        # maxpool,332*360*640 --> 32*180*320
        self.pool1=nn.MaxPool2d(2,2)
    
        # layer2, 32*180*320 --> 64*180*320 
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        # maxpool,64*180*320 --> 64*90*160
        self.pool2=nn.MaxPool2d(2,2)
    
        # layer3, 64*90*160 --> 128*90*160 
        self.conv3=nn.Conv2d(64,128,kernel_size=1,padding=0)
    
        # layer4, 128*90*160  --> 64*90*160
        self.conv4=nn.Conv2d(128,64,kernel_size=3,padding=1)
        # maxpool,64*90*160 --> 64*45*80
        self.pool4=nn.MaxPool2d(2,2)
        
        # layer5, 64*45*80 --> 128*45*80
        self.conv5=nn.Conv2d(64,128,kernel_size=3,padding=1)
        # maxpool,128*45*80 --> 128*23*40
        self.pool5=nn.MaxPool2d(2,2,padding=[1,0])
    
        # layer6, 128*23*40 --> 256*23*40 
        self.conv6=nn.Conv2d(128,256,kernel_size=3,padding=1)
        # maxpool,256*23*40 --> 256*12*20
        self.pool6=nn.MaxPool2d(2,2,padding=[1,0])
        
        # layer7, 256*12*20 --> 128*12*20 
        self.conv7=nn.Conv2d(256,128,kernel_size=1,padding=0)
        
        # layer8, 128*12*20 --> 256*12*20 
        self.conv8=nn.Conv2d(128,256,kernel_size=3,padding=1)
        # maxpool,256*12*20 --> 256*6*10
        self.pool8=nn.MaxPool2d(2,2)
        
        # layer9, 256*6*10 --> 512*6*10 
        self.conv9=nn.Conv2d(256,512,kernel_size=3,padding=1)
        # maxpool,512*6*10 --> 512*3*5
        self.pool9=nn.MaxPool2d(2,2)
        
        # layer10, 512*40*22 --> 256*40*22
        self.conv10=nn.Conv2d(512,256,kernel_size=1,padding=0)
        
        # layer11, 256*3*5 --> 512
        self.conv11=nn.Conv2d(256,512,kernel_size=[3,5],padding=0)
        
        # layer12, 512 --> 96
        self.line12=nn.Linear(512,96,bias=True)
        
        # layer13, 96 --> 12
        self.line13=nn.Linear(96,12,bias=True)
        
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
        x=self.pool4(x)
        
        # layer5,activation=leaky
        x=self.conv5(x)
        x=F.leaky_relu(x)
        x=self.pool5(x)

        # layer6,activation=leaky
        x=self.conv6(x)
        x=F.leaky_relu(x)
        x=self.pool6(x)
        
        # layer7,activation=leaky
        x=self.conv7(x)
        x=F.leaky_relu(x)
        
        # layer8,activation=leaky
        x=self.conv8(x)
        x=F.leaky_relu(x)
        x=self.pool8(x)
        
        # layer9,activation=leaky
        x=self.conv9(x)
        x=F.leaky_relu(x)
        x=self.pool9(x)
        
        # layer10,activation=leaky
        x=self.conv10(x)
        x=F.leaky_relu(x)
        
        # layer11,activation=leaky
        x=self.conv11(x)
        x=F.leaky_relu(x)
        x=x.view(-1,512)
        
        # layer12
        x=self.line12(x)
        x=F.leaky_relu(x)
        
        # layer13
        x=self.line13(x)
        x=F.softmax(x,dim=1)
        
        
        return(x)
    
    
    
class simplenet(nn.Module):
    
    def __init__(self):
        super(simplenet,self).__init__()
    
        # layer1, 3*640*360  --> 32*640*360 
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        # maxpool,32*640*360 --> 32*320*180
        self.pool1=nn.MaxPool2d(2,2)
    
        # layer2, 32*320*180 --> 64*320*180 
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        # maxpool,64*320*180 --> 64*160*90
        self.pool2=nn.MaxPool2d(2,2)
    
        # layer3, 64*160*90 --> 128*160*90 
        self.conv3=nn.Conv2d(64,128,kernel_size=1,padding=0)
    
        # layer4, 128*160*90 --> 64*160*90 
        self.conv4=nn.Conv2d(128,64,kernel_size=3,padding=1)
    
        # layer5, 64*160*90 --> 128*160*90 
        self.conv5=nn.Conv2d(64,128,kernel_size=3,padding=1)
        # maxpool,128*160*90 --> 128*80*45
        self.pool5=nn.MaxPool2d(2,2)
    
        # layer6, 128*80*45 --> 256*80*45 
        self.conv6=nn.Conv2d(128,256,kernel_size=3,padding=1)
        
        # layer7, 256*80*45 --> 128*80*45 
        self.conv7=nn.Conv2d(256,128,kernel_size=1,padding=0)
        
        # layer8, 128*80*45 --> 256*80*45 
        self.conv8=nn.Conv2d(128,256,kernel_size=3,padding=1)
        # maxpool,256*80*45 --> 256*40*22.5?
        self.pool8=nn.MaxPool2d(2,2)
        
        # layer9, 256*40*22 --> 512*40*22 
        self.conv9=nn.Conv2d(256,512,kernel_size=3,padding=1)
        
        # layer10, 512*40*22 --> 256*40*22
        self.conv10=nn.Conv2d(512,256,kernel_size=1,padding=0)
        
        # layer11, 256*40*22 --> 512*40*22 
        self.conv11=nn.Conv2d(256,512,kernel_size=3,padding=1)
        
        # layer12, 512*40*22 --> 256*40*22
        self.conv12=nn.Conv2d(512,256,kernel_size=1,padding=0)
        
        # layer13, 256*40*22 --> 512*40*22 
        self.conv13=nn.Conv2d(256,512,kernel_size=3,padding=1)
        # maxpool, 512*40*22 --> 512*20*11
        self.pool13=nn.MaxPool2d(2,2)
        
        # layer14, 512*20*11 --> 1024**20*11
        self.conv14=nn.Conv2d(512,1024,kernel_size=3,padding=1)
        
        # layer15, 1024*20*11 --> 512*20*11
        self.conv15=nn.Conv2d(1024,512,kernel_size=1,padding=0)
        
        # layer16, 512*20*11 --> 1024*20*11
        self.conv16=nn.Conv2d(512,1024,kernel_size=3,padding=1)
        
        # layer17, 1024*20*11 --> 512*20*11
        self.conv17=nn.Conv2d(1024,512,kernel_size=1,padding=0)
        
        # layer18, 512*20*11 --> 1024
        self.conv18=nn.Conv2d(512,1024,kernel_size=3,padding=1)
        self.pool18=nn.MaxPool2d((11,20),stride=1)
        
        # layer19, 1024 --> 96
        self.line19=nn.Linear(1024,96,bias=True)
        
        # layer20, 96 --> 12
        self.line20=nn.Linear(96,12,bias=True)
        
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
        x=self.pool5(x)

        # layer6,activation=leaky
        x=self.conv6(x)
        x=F.leaky_relu(x)
        
        # layer7,activation=leaky
        x=self.conv7(x)
        x=F.leaky_relu(x)
        
        # layer8,activation=leaky
        x=self.conv8(x)
        x=F.leaky_relu(x)
        x=self.pool8(x)
        
        # layer9,activation=leaky
        x=self.conv9(x)
        x=F.leaky_relu(x)
        
        # layer10,activation=leaky
        x=self.conv10(x)
        x=F.leaky_relu(x)
        
        # layer11,activation=leaky
        x=self.conv11(x)
        x=F.leaky_relu(x)
        
        # layer12,activation=leaky
        x=self.conv12(x)
        x=F.leaky_relu(x)
        
        # layer13,activation=leaky
        x=self.conv13(x)
        x=F.leaky_relu(x)
        x=self.pool13(x)
        
        # layer14,activation=leaky
        x=self.conv14(x)
        x=F.leaky_relu(x)
        
        # layer15,activation=leaky
        x=self.conv15(x)
        x=F.leaky_relu(x)
        
        # layer16,activation=leaky
        x=self.conv16(x)
        x=F.leaky_relu(x)
        
        # layer17,activation=leaky
        x=self.conv17(x)
        x=F.leaky_relu(x)
        
        # layer18,activation=leaky
        x=self.conv18(x)
        x=F.leaky_relu(x)
        x=self.pool18(x)
        x=x.view(-1,1024)
        
        # layer19
        x=self.line19(x)
        x=F.leaky_relu(x)
        
        # layer20
        x=self.line20(x)
        #x=x.view(-1,12)
        x=F.softmax(x,dim=1)
        
        
        return(x)
