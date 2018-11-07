from torchvision import models
import torch
from torch import nn
import math

class fast_rcnn_net(nn.Module):

    def __init__(self):

        # TODO: use vgg16 as ConvNet
        self.features = nn.Sequential(
            # 0-0 conv layer: 360 * 640 * 64
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLu(),

            # 0-1 conv layer: 360 * 640 * 64
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 0 max pooling: 180 * 320 * 64
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # 1-0 conv layer: 180 * 320 * 128
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 1-1 conv layer: 180 * 320 * 128
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 1 max pooling: 90 * 160 * 128
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # 2-0 conv layer: 90 * 160 * 256
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 2-1 conv layer: 90 * 160 * 256
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 2-2 conv layer: 90 * 160 * 256
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 2 max pooling: 45 * 80 * 256
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # 3-0 conv layer: 45 * 80 * 512
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        # according to the fast-rcnn paper, it is unnecessary to change the parameters of the first 8 conv layers, thus we freeze these conv layers so the parameters won't be updated during training
        for param in self.parameters():
            param.requires_grad = False     # freeze these parameters

        # from here on, the parameters will be updated by back-propagation
        self.features_unfreeze = nn.Sequential(
            # 3-1 conv layer: 45 * 80 * 512
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 3-2 conv layer: 45 * 80 * 512
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 3 max pooling: 22 * 40 * 512
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # 4-0 conv layer: 22 * 40 * 512
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 4-1 conv layer: 22 * 40 * 512
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 4-2 conv layer: 22 * 40 * 512
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        # TODO: the ROI Pooling layer
        # the last max pooling layer is replaced by a ROI pooling layer that is configured by setting H=W=7: 7 * 7 * 512
        self.roi_pooling = nn.AdaptiveMaxPool2d((7, 7), return_indices=False)

        # TODO: continue the vgg fully connected layer
        self.classifier = nn.Sequential(
            # 0 fully connected
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # 1 fully connected
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # TODO: two sibling output layer: one that produces softmax probability estimates, another outputs four real-valued numbers for each of the object class


    def forward(self, x, region_projs):
        output = self.features(x)
        output = self.features_unfreeze(output)

        output = self.roi_pooling(output)


        return output


def map_region_proposals_to_feature_map (rps):
    rp_projs = []
    for rp in rps:
        (r1, c1, w, h) = rp  # (r1, c1) is the top-left corner of the region proposal, w is width, h is height
        r2, c2 = r1 + w - 1, c1 - h + 1  # (r2, c2) is the bottom-right corner

        r1_ = math.floor(r1 / 16) + 1
        c1_ = math.floor(c1 / 16) + 1
        r2_ = math.floor(r2 / 16) - 1
        c2_ = math.floor(c2 / 16) - 1
        rp_projs.append((r1_, c1_, r2_-r1_+1, c1_-c2_+1))
    return rp_projs


def get_fast_rcnn_net ():
    our_net = fast_rcnn_net()

    # load the vgg16 pre-trained parameter values
    pretrained_vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = pretrained_vgg16.state_dict()

    # update our network(the vgg16 part) with pre-trained vgg16 parameter values
    our_net_dict = our_net.state_dict()
    pretrained_dict = dict({k: v for k, v in pretrained_dict.items() if k in our_net_dict})
    our_net_dict.update(pretrained_dict)
    our_net.load_state_dict(our_net_dict)

    return our_net

def smooth_multi_task_loss (scores, bs, region_num):
    pass