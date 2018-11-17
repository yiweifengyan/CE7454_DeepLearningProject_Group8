import torch
import numpy as np
import time
import fast_rcnn_network_cuda
import readimg_new
import os
from torchvision import models
from fast_rcnn_network_cuda import fast_rcnn_net
from datetime import datetime

root_path = os.getcwd()
segment = "segment2"

device = torch.device("cuda")
model_name = ''.join(str(datetime.now())[11:19].split(':'))
print(model_name)

# get data
# region_proposal: r, c, w, h, label; (r, c) is the axis of left-top corner, label is 0(background) if IOU<50% else is the ground truth label
train_data, train_label, train_rps_4_imgs, train_rp_labels_4_imgs = readimg_new.read_data_with_rps(
    [segment + "/train_data", segment + "/train_label", segment + "/region_proposals_train"])

train_rp_labels_4_imgs = torch.Tensor(train_rp_labels_4_imgs)

train_data = train_data.type(torch.float).to(device)
train_label = train_label.type(torch.float).to(device)

# map region proposals to feature maps
train_rg_projs_4_imgs = []
for rps_4_img in train_rps_4_imgs:
    train_rg_projs_4_imgs.append(fast_rcnn_network_cuda.map_region_proposals_to_feature_map(rps_4_img))
train_rg_projs_4_imgs = torch.Tensor(train_rg_projs_4_imgs)

# fast rcnn network
output_size = 16
our_net = fast_rcnn_net(output_size).to(device)

# load the vgg16 pre-trained parameter values
pretrained_vgg16 = models.vgg16(pretrained=True)
pretrained_dict = pretrained_vgg16.state_dict()

# update our network(the vgg16 part) with pre-trained vgg16 parameter values
our_net_dict = our_net.state_dict()
pretrained_dict = dict({k: v for k, v in pretrained_dict.items() if k in our_net_dict})
our_net_dict.update(pretrained_dict)
our_net.load_state_dict(our_net_dict)

# TODO: train net

train_data_num = train_data.size()[0]
#test_data_num = test_data.size()[0]
pixel_num_of_each_pict = np.prod(train_data[0].size()[-2:])
start_time = time.time()
epoch_num = 20
bs = 2  # number of images in each batch, as mentioned in the paper
lr = 0.05
train_region_num = 32  # number of region_projections for each train image
sub_bs = 16

print("epoch_num=", epoch_num, ", bs=", bs, ", lr=", lr)

for epoch in range(epoch_num):
    print("epoch: ", epoch)
    # learning rate strategy : divide the learning rate by 1.5 every 10 epochs
    if epoch % 10 == 0 and epoch > 0:
        lr /= 1.5

    # create a new optimizer at the beginning of each epoch: give the current learning rate
    optimizer = torch.optim.SGD(our_net.parameters(), lr=lr)

    shuffled_indices = torch.randperm(train_data_num)

    running_loss_train = 0
    num_train = 0

    for count in range(0, train_data_num, bs):
        print("count:", count)
        # forward and backward pass
        optimizer.zero_grad()

        indices = shuffled_indices[count: count + bs]
        minibatch_train_data = train_data[indices]
        minibatch_train_label = train_label[indices]
        minibatch_train_rp_label = train_rp_labels_4_imgs[indices]
        minibatch_train_rg_projs = train_rg_projs_4_imgs[indices]
        train_inputs = minibatch_train_data

        train_inputs.requires_grad_()

        feature_maps = our_net.forward_feature(train_inputs)

        loss = 0
        for i in range(train_region_num):
            #print("i:", i)
            region_projs = [minibatch_train_rg_projs[j][i] for j in range(min(bs, len(minibatch_train_rg_projs)))]  # the ith region for the jth image
            rp_labels = [minibatch_train_rp_label[j][i] for j in range(min(bs, len(minibatch_train_rp_label)))]

            clf_scores, bbox_pred = our_net.forward_output(feature_maps, region_projs)
            clf_gtruth = [z[0] for z in minibatch_train_label]
            bbox_gtruth = [z[1:] for z in minibatch_train_label]
            loss += fast_rcnn_network_cuda.smooth_multi_task_loss(clf_scores, clf_gtruth, bbox_pred, bbox_gtruth, rp_labels, 1)

            running_loss_train += loss.detach().item()
            num_train += 1
            count_train_loss += loss.detach().item()

        print("count=", count, ", total train loss=", count_train_loss, ", lr=", lr, ", ", minibatch_train_label)
        loss.backward()
        optimizer.step()

    total_loss_train = round(running_loss_train / num_train, 4)
    elapsed_time = round(time.time() - start_time, 4)
    print("epoch=", epoch, ", time=", elapsed_time, ", train loss=", total_loss_train, ", lr=", lr)


# save the model parameters
torch.save(our_net.state_dict(), segment + "/model_params_" + model_name + ".pkl")