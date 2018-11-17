from torchvision import models
import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np

stride_prod = 16
roi_size = 7
roi_pad = 0

class fast_rcnn_net(nn.Module):

    def __init__(self, output_size):
        super(fast_rcnn_net, self).__init__()

        # TODO: use vgg16 as ConvNet
        self.features = nn.Sequential(
            # 0-0 conv layer: 3 * 360 * 640 -> 64 * 360 * 640
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 0-1 conv layer: 64 * 360 * 640 -> 64 * 360 * 640
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 0 max pooling: 64 * 360 * 640 -> 64 * 180 * 320
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # 1-0 conv layer: 64 * 180 * 320 -> 128 * 180 * 320
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 1-1 conv layer: 128 * 180 * 320 -> 128 * 180 * 320
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 1 max pooling: 128 * 180 * 320 -> 128 * 90 * 160
            nn.MaxPool2d(kernel_size=2, stride =2, padding=0, dilation=1, ceil_mode=False),

            # 2-0 conv layer: 128 * 90 * 160 -> 256 * 90 * 160
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 2-1 conv layer: 256 * 90 * 160 -> 256 * 90 * 160
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 2-2 conv layer: 256 * 90 * 160 -> 256 * 90 * 160
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 2 max pooling: 256 * 90 * 160 -> 256 * 45 * 80
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # 3-0 conv layer: 256 * 45 * 80 -> 512 * 45 * 80
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()

        )

        # according to the fast-rcnn paper, it is unnecessary to change the parameters of the first 8 conv layers, thus we freeze these conv layers so the parameters won't be updated during training
        for param in self.parameters():
            param.requires_grad = False     # freeze these parameters
        
        # from here on, the parameters will be updated by back-propagation
        self.features_unfreeze = nn.Sequential(
            # 3-1 conv layer: 512 * 45 * 80 -> 512 * 45 * 80
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 3-2 conv layer: 512 * 45 * 80 -> 512 * 45 * 80
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 3 max pooling: 512 * 45 * 80 -> 512 * 22 * 40
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # 4-0 conv layer: 512 * 22 * 40 -> 512 * 22 * 40
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 4-1 conv layer: 512 * 22 * 40 -> 512 * 22 * 40
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),

            # 4-2 conv layer: 512 * 22 * 40 -> 512 * 22 * 40
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        # TODO: the ROI Pooling layer
        # the last max pooling layer is replaced by a ROI pooling layer that is configured by setting H=W=roi_size: roi_size * roi_size * 512
        self.roi_pooling = nn.AdaptiveMaxPool2d((roi_size, roi_size), return_indices=False)

        # TODO: continue the vgg fully connected layer
        self.classifier = nn.Sequential(
            # 0 fully connected
            nn.Linear(in_features=roi_size*roi_size*512, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # 1 fully connected
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # TODO: two sibling output layer: one that produces softmax probability estimates, another outputs four real-valued numbers for each of the object class
        self.class_score_layer = nn.Linear(in_features=4096, out_features=output_size, bias=False)

        self.bbox_target_layer = nn.Linear(in_features=4096, out_features=(output_size-1)*4, bias=False)

    ## bbox regressor uses the parameterization for regression targets given in paper "Rich feature hierarchies for accurate object detection and semantic segmentation"
    def bbox_target_to_pred_bbox(self, region_proj, bbox_target):
        box = torch.Tensor(region_proj)

        r, c, w, h = box[0], box[1], box[2], box[3]

        dr = bbox_target[0::4]
        dc = bbox_target[1::4]
        dw = bbox_target[2::4]
        dh = bbox_target[3::4]

        pred_bbox = torch.zeros(bbox_target.size(), dtype=bbox_target.dtype)

        pred_bbox[0::4] = w * dr + r
        pred_bbox[1::4] = h * dc + c
        pred_bbox[2::4] = w * torch.Tensor(np.exp(dw.detach()))
        pred_bbox[3::4] = h * torch.Tensor(np.exp(dh.detach()))

        for i in range(len(pred_bbox.detach())):
            if i % 4 == 0 or i % 4 == 1:
                pred_bbox[i] = math.ceil(pred_bbox[i] * stride_prod) - 1
            if i % 4 == 2 or i % 4 == 3:
                pred_bbox[i] = math.floor(pred_bbox[i] * stride_prod) + 1

        return pred_bbox

    def forward_feature (self, x):
        feature_maps = self.features(x)
        #feature_maps = self.features_unfreeze(feature_maps)
        return feature_maps

    def forward_output (self, x, region_projs):
        size = x.detach().size()
        output = torch.Tensor(size[0], size[1], roi_size, roi_size)
        for idx in range(size[0]):
            (r, c, w, h) = (int(z) for z in region_projs[idx])
            output[idx] = self.roi_pooling(F.pad(x[idx, :, c: c+h, r: r+w], (roi_pad, roi_pad, roi_pad, roi_pad)))
        output = self.classifier(output.view(size[0], -1))
        clf_scores = self.class_score_layer(output)
        clf_scores = F.softmax(clf_scores, dim=1)
        bbox_targets = self.bbox_target_layer(output)
        bbox_pred = torch.Tensor(bbox_targets.detach().size())
        for idx in range(len(region_projs)):
            bbox_pred[idx] = self.bbox_target_to_pred_bbox(region_projs[idx], bbox_targets[idx])
        return clf_scores, bbox_pred
        return clf_scores, bbox_targets


def map_region_proposals_to_feature_map (rps):
    rp_projs = []
    for rp in rps:
        (r1, c1, w, h) = rp  # (r1, c1) is the top-left corner of the region proposal, w is width, h is height
        r2, c2 = r1 + w - 1, c1 + h - 1  # (r2, c2) is the bottom-right corner

        r1_ = min(math.floor(r1 / stride_prod) + 1, 38)  # max index is 39, but we have to guarantee that the projection has at least width 1
        c1_ = min(math.floor(c1 / stride_prod) + 1, 20)  # max index is 21, ...
        r2_ = math.ceil(r2 / stride_prod) - 1
        c2_ = math.ceil(c2 / stride_prod) - 1
        w = max(1.0, r2_-r1_+1)
        h = max(1.0, c2_-c1_+1)
        rp_projs.append((r1_, c1_, w, h))
    return rp_projs


def smooth_multi_task_loss (clf_scores, clf_gtruth, bbox_pred, bbox_gtruth, bbox_label, lambda_):
    loss = torch.zeros(len(clf_gtruth))
    #criterion = nn.CrossEntropyLoss()
    for idx in range(len(clf_gtruth)):
        loss_cls = torch.Tensor([- math.log(max(clf_scores[idx][int(clf_gtruth[idx].item())], 1e-45))]).squeeze(0)
        loss_cls.requires_grad_()
        u = int(bbox_label[idx].item())
        if u > 0:
            loss_bbox = F.smooth_l1_loss(bbox_pred[idx][(u - 1)*4: u*4], bbox_gtruth[idx].type(torch.float), reduction="sum")
        else:
            loss_bbox = 0

        loss[idx] = loss_cls + lambda_ * loss_bbox

    return loss.mean(dim=0)