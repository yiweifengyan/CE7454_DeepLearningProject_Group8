import torch
import torch.nn as nn
import numpy as np
import utils
import time
import torch.nn.functional as F

def train (train_data, train_label, test_data, test_label, net, optim, bs=200, criterion=None, lr=0.05, epoch_num=30):

    if criterion == None:
        criterion = nn.CrossEntropyLoss()

    train_data_num = train_data.size()[0]
    test_data_num = test_data.size()[0]
    pixel_num_of_each_pict = np.prod(train_data[0].size()[-2:])
    start_time = time.time()

    def eval_on_test_set():
        running_error = 0
        num_batches = 0

        for i in range(0, test_data_num, bs):
            minibatch_test_data = test_data[i: i+bs]
            minibatch_test_label = test_label[i: i+bs]
            test_inputs = minibatch_test_data.view(bs, 3 * pixel_num_of_each_pict)
            test_scores = net(test_inputs)
            error = utils.get_error(test_scores, minibatch_test_label)
            running_error += error.item()
            num_batches += 1

        total_error = running_error / num_batches
        return total_error


    for epoch in range(epoch_num):
        # learning rate strategy : divide the learning rate by 1.5 every 10 epochs
        if epoch % 10 == 0 and epoch > 10:
            lr /= 1.5

        # create a new optimizer at the beginning of each epoch: give the current learning rate
        optimizer = optim(net.parameters(), lr=lr)

        shuffled_indices = torch.randperm(train_data_num)

        for count in range(0, train_data_num, bs):
            # forward and backward pass
            optimizer.zero_grad()

            indices = shuffled_indices[count: count+bs]
            minibatch_train_data = train_data[indices]
            print(minibatch_train_data.size())
            minibatch_train_label = train_label[indices]

            train_inputs = minibatch_train_data.view(bs, 3 * pixel_num_of_each_pict)

            train_inputs.requires_grad_()

            scores = net(train_inputs)

            loss = criterion(scores, minibatch_train_label)

            loss.backward()

            optimizer.step()

        # compute the error rate after each epoch
        elapsed_time = time.time() - start_time
        print("epoch=", epoch, ", time=", elapsed_time, ", error=", eval_on_test_set()*100, "%, lr=", lr)

    return net

# test here
import readimg_new

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
net = train (train_data, train_label, test_data, test_label, net, torch.optim.SGD, bs=20)

list_of_param = list(net.parameters())
print(list_of_param)