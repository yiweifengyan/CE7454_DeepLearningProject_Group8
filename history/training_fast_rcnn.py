import torch
import numpy as np
import utils
import time
import fast_rcnn_network
import readimg_new


# get data
train_data, train_label = readimg_new.read_data(["train_data", "train_label"])
test_data, test_label = readimg_new.read_data(["test_data", "test_label"])

# get region proposals
region_proposals_4_all_images = []

# map region proposals to feature maps
region_projs_4_all_images = []
for rps_4_one_image in region_proposals_4_all_images:
    region_projs_4_all_images.append(fast_rcnn_network.map_region_proposals_to_feature_map(rps_4_one_image))

# fast rcnn network
our_net = fast_rcnn_network.get_fast_rcnn_net()

# TODO: train net

train_data_num = train_data.size()[0]
test_data_num = test_data.size()[0]
pixel_num_of_each_pict = np.prod(train_data[0].size()[-2:])
start_time = time.time()
epoch_num = 30
bs = 2 # number of images in each batch, as mentioned in the paper
lr = 0.05
region_num = 64 # number of region_projections for each image

def eval_on_test_set(net, bs):
    running_error_test = 0
    num_batches_test = 0

    for i in range(0, test_data_num, bs):
        minibatch_test_data = test_data[i: i + bs]
        minibatch_test_label = test_label[i: i + bs]
        test_inputs = minibatch_test_data.view(-1, 3 * pixel_num_of_each_pict)
        test_scores = net(test_inputs)
        error = utils.get_error(test_scores, minibatch_test_label)

        running_error_test += error.item()
        num_batches_test += 1

    total_error_test = round(running_error_test / num_batches_test, 4)
    return total_error_test


for epoch in range(epoch_num):
    # learning rate strategy : divide the learning rate by 1.5 every 10 epochs
    if epoch % 10 == 0 and epoch > 0:
        lr /= 1.5

    # create a new optimizer at the beginning of each epoch: give the current learning rate
    optimizer = torch.optim.SGD(our_net.parameters(), lr=lr)

    shuffled_indices = torch.randperm(train_data_num)

    running_loss_train = 0
    running_error_train = 0
    num_batches_train = 0

    for count in range(0, train_data_num, bs):
        # forward and backward pass
        optimizer.zero_grad()

        indices = shuffled_indices[count: count + bs]
        minibatch_train_data = train_data[indices]
        minibatch_train_label = train_label[indices]
        train_inputs = minibatch_train_data.view(-1, 3 * pixel_num_of_each_pict)
        train_inputs.requires_grad_()

        scores = our_net(train_inputs, region_projs_4_all_images[indices])
        loss = fast_rcnn_network.smooth_multi_task_loss(scores, minibatch_train_label, bs, region_num)
        loss.backward()
        optimizer.step()

        running_loss_train += loss.detach().item()
        error = utils.get_error(scores.detach(), minibatch_train_label)
        running_error_train += error.item()
        num_batches_train += 1

    total_loss_train = round(running_loss_train / num_batches_train, 4)
    total_error_train = round(running_error_train / num_batches_train, 4)
    elapsed_time = round(time.time() - start_time, 4)
    print("epoch=", epoch, ", time=", elapsed_time, ", train loss=", total_loss_train, ", train error=", total_error_train, ", test error=", eval_on_test_set(), ", lr=", lr)
