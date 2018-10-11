#This is the YOLO version of the project
#The architecture of the neural network
#Version 0.0 by Shuan, 2018-10-11
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import time
import utils

class simplynet():
  def _init_(self):
    super(simplynet,self)._init_()
    
    self.conv1=nn.Conv2d(3,64,kernel_size=3,padding=1)
