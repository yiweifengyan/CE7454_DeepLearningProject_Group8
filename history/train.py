import torch
import torch.nn as nn
import numpy as np
import utils
import time
import torch.nn.functional as F

def train (train_data, train_label, test_data, test_label, net, bs=200, criterion=None, lr=0.05, epoch_num=30):

    if criterion == None:
        criterion = nn.CrossEntropyLoss()

    train_data_num = train_data.size()[0]
    test_data_num = test_data.size()[0]
    pixel_num_of_each_pict = np.prod(train_data[0].size()[-2:])
    start_time = time.time()

    def eval_on_test_set():
        running_error_test = 0
        num_batches_test = 0

        for i in range(0, test_data_num, bs):
            minibatch_test_data = test_data[i: i+bs]
            minibatch_test_label = test_label[i: i+bs]
            test_inputs = minibatch_test_data.view(-1, 3 * pixel_num_of_each_pict)
            test_scores = net(test_inputs)
            error = utils.get_error(test_scores, minibatch_test_label)

            running_error_test += error.item()
            num_batches_test += 1

        total_error_test = round(running_error_test / num_batches_test, 4)
        return total_error_test


    for epoch in range(epoch_num):
        # learning rate strategy : divide the learning rate by 1.5 every 10 epochs
        if epoch % 5 == 10 and epoch > 10:
            lr /= 1.5

        # create a new optimizer at the beginning of each epoch: give the current learning rate
        optimizer = torch.optim.SGD (net.parameters(), lr=lr)

        shuffled_indices = torch.randperm(train_data_num)

        running_loss_train = 0
        running_error_train = 0
        num_batches_train = 0

        for count in range(0, train_data_num, bs):
            # forward and backward pass
            optimizer.zero_grad()

            indices = shuffled_indices[count: count+bs]
            minibatch_train_data = train_data[indices]
            minibatch_train_label = train_label[indices]
            train_inputs = minibatch_train_data.view(-1, 3 * pixel_num_of_each_pict)
            train_inputs.requires_grad_()
            scores = net(train_inputs)
            loss = criterion(scores, minibatch_train_label)
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

    return net



# test the train function here
import readimg_new
# do the label classification job
train_data, train_label = readimg_new.read_data(["train_data", "train_label"])
test_data, test_label = readimg_new.read_data(["test_data", "test_label"])
train_label = train_label[:, 0]
test_label = test_label[:, 0]

# test with one_layer_net
class one_layer_net(nn.Module):
    def __init__(self, input_size, output_size):
        super(one_layer_net, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size, bias=False)
    def forward(self, x):
        scores = self.layer1(x)
        return scores

net = one_layer_net(3 * 360 * 640, 13)
net = train (train_data, train_label, test_data, test_label, net, bs=20)
list_of_param = list(net.parameters())
print(list_of_param)

import inference

inference.inference(net, test_data[0])
inference.inference(net, test_data[10])
inference.inference(net, test_data[15])
inference.inference(net, test_data[20])
inference.inference(net, test_data[25])
