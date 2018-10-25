#This is the YOLOv2 version of the project
#The architecture of the neural network
#Version 0.0 by Shuan, 2018-10-11
#Finished first 18 layers of YOLOv2 by Shaun, 2018-10-12
#Made it a simple net only for classification, by Shaun, 2018-10-19
#Fixed some bugs, by Shaun, 2018-10-25

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import time
import utils


#define the Neural Network
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

    
# The codes below are for small scale testing
bs=20
x=torch.rand(bs,3,360,640)
print(x.dim())
utils.show(x[0])
testnet=simplynet()
print(x.size())
x=testnet(x)
print(x.size())
#print(x)
utils.show_prob(x[0])

# The codes below are for small scale training, it is useful
import readimg_new
train_data, train_label = readimg_new.read_data(["train_data", "train_label"])
test_data, test_label = readimg_new.read_data(["test_data", "test_label"])
train_label = train_label[:, 0]
print(train_label.type())
test_label = test_label[:, 0]

device= torch.device("cuda")
print(device)

net = simplynet()
#net=net.to(device)
bs=5
#net = train_simplynet(train_data, train_label, test_data, test_label, net, torch.optim.SGD, bs)
criterion = nn.NLLLoss()
optimizer=torch.optim.SGD(net.parameters() , lr=0.01 )

for iter in range(1,200):
    
    # create a minibatch
    indices=torch.LongTensor(bs).random_(0,505)
    minibatch_data = train_data[indices]
    minibatch_label= train_label[indices]
    print(minibatch_label.type())
    # feed the input to the net  
    inputs=minibatch_data
    inputs.requires_grad_()
    prob=net(inputs) 
   
    # update the weights (all the magic happens here -- we will discuss it later)
    log_prob=torch.log(prob)
    loss = criterion(log_prob, minibatch_label)    
    optimizer.zero_grad()       
    loss.backward()
    optimizer.step()
    
x=test_data[100:105]
print(x.size())
x=net(x)
print(x.size())
for i in range(5):
    utils.show_prob(x[i])
