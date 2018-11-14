import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import numpy as np
import utils
import os
import time
import readin
import mynet

def get_accu(out_prob,real_label,bs):
    count=0
    for i in range(bs):
        temp=0.0
        maxp=0
        for j in range(12):
            if temp<out_prob[i,j]:
                temp=out_prob[i,j]
                maxp=j
        if maxp==real_label[i]:
            count=count+1
    return(count)

mydir=os.getcwd()
train_data_dir=mydir+"/train_data"
train_label_dir=mydir+"/train_label"
small_data_dir=mydir+"/small_data"
small_label_dir=mydir+"/small_label"
device= torch.device("cuda:0")
print(device)

net = mynet.simplynet_v3()
net=net.to(device)
origname="simplynet_V3"
record=open(origname+"_record.txt",mode='a')

bs=20
lr=0.01
epoch_num=20
train_data_num=1500
iter_num=int(train_data_num/bs)
criterion = nn.NLLLoss()
optimizer=torch.optim.SGD(net.parameters(),lr)

start_time = time.time()
for epoch in range(epoch_num):
    if (epoch>0)and(epoch%4==0) :
        del optimizer
        lr=lr/2
    optimizer=torch.optim.SGD(net.parameters(),lr)
    
    for step in range(int(12000/train_data_num)):
        # read in the data
        train_data = readin.read_part_image(train_data_dir,step*train_data_num,train_data_num)
        train_label= readin.read_part_xml(train_label_dir,step*train_data_num,train_data_num)
        print(train_data.size(),train_data.type())
        train_label=train_label[:,0]
        train_data=train_data.to(device)
        train_label=train_label.to(device)
        
        step_loss=0.0
        for iter in range(iter_num):
            if iter>0 :
                del indices
                del minibatch_data
                del minibatch_label
                del inputs
                del prob
                del log_prob
                del loss
                
            # create a minibatch
            indices=iter*bs
            minibatch_data = train_data[indices:indices+bs]
            minibatch_label= train_label[indices:indices+bs]
            minibatch_data=minibatch_data.to(device)
            minibatch_label=minibatch_label.to(device)
    
            # feed the input to the net
            inputs=minibatch_data
            inputs.requires_grad_()
            prob=net(inputs)
    
            # update the weights
            log_prob=torch.log(prob)
            loss=criterion(log_prob, minibatch_label)
            #loss=criterion(prob, minibatch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_loss=step_loss+loss.item()
            record.write("\nbatch no."+str(step*iter_num+iter)+" loss: "+'{0:.4f}'.format(loss.item())+" time: "+'{0:.2f}'.format(time.time()-start_time))
            
    del train_data
    del train_label
    del indices
    del minibatch_data
    del minibatch_label
    del inputs
    del prob
    del log_prob
    del loss
    print("step loss: ",step_loss/iter_num)
    record.write("\nstep loss: "+'{0:.4f}'.format(step_loss/iter_num))
        
    small_data = readin.read_part_image(small_data_dir,0,1740)
    small_label= readin.read_part_xml(small_label_dir,0,1740)
    small_label=small_label[:,0]
    small_data=small_data.to(device)
    small_label=small_label.to(device)
    small_loss=0.0
    accu_count=0
    for iter in range(int(1740/bs)):
        if iter>0 :
            del indices
            del minibatch_data
            del minibatch_label
            del inputs
            del prob
            del log_prob
            del loss
                
        # create a minibatch
        indices=iter*bs
        minibatch_data = small_data[indices: indices+bs]
        minibatch_label= small_label[indices: indices+bs]
        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        #print(minibatch_label.type())
    
        # feed the input to the net
        inputs=minibatch_data
        prob=net(inputs)
        
        # get the accuracy
        accu_count=accu_count+get_accu(prob,minibatch_label,bs)
        
        # get the loss
        log_prob=torch.log(prob)
        loss=criterion(log_prob, minibatch_label)
        #loss=criterion(prob, minibatch_label)
        small_loss=small_loss+loss.item()
             
    del indices
    del minibatch_data
    del minibatch_label
    del inputs
    del prob
    del loss   
    del small_data
    del small_label
    print("small loss:",small_loss/87,"accuracy:",accu_count/1740)
    record.write("small loss:"+'{0:.4f}'.format(small_loss/87)+" accuracy:"+'{0:.4f}'.format(accu_count/1740))
    
    net_name=origname+"_epoch_"+str(epoch)+".pt"
    torch.save(net.state_dict(),net_name)
record.close()


