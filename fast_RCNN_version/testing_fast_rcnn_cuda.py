import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import fast_rcnn_network
from fast_rcnn_network import fast_rcnn_net
import readimg_new
import numpy as np
import utils
import os
from datetime import datetime
import training_fast_rcnn_cuda as train

root_path = os.getcwd() + "\\"

segment = "segment-random"
model = "model_params.pkl"

device = torch.device("cuda")
our_net = fast_rcnn_net(train.output_size)
our_net.load_state_dict(torch.load(root_path + segment + "\\" + model))

our_net = our_net.to(device)

test_data, test_label, test_rps_4_imgs, test_rp_labels_4_imgs = readimg_new.read_data(
    [segment + "\\test_data", segment + "\\test_label"])

test_data = test_data.to(device)
test_label = test_label.to(device)

test_rg_projs_4_imgs = []
for rps_4_img in test_rps_4_imgs:
    test_rg_projs_4_imgs.append(fast_rcnn_network.map_region_proposals_to_feature_map(rps_4_img))

def get_label_by_number (no_):
    if no_ == 1:
        return "boat1"
    if no_ == 2:
        return "boat2"
    if no_ == 3:
        return "boat3"
    if no_ == 4:
        return "boat5"
    if no_ == 5:
        return "car1"
    if no_ == 6:
        return "car12"
    if no_ == 7:
        return "car17"
    if no_ == 8:
        return "car20"
    if no_ == 9:
        return "person11"
    if no_ == 10:
        return "person19"
    if no_ == 11:
        return "person22"
    if no_ == 12:
        return "person29"
    if no_ == 13:
        return "riding3"
    if no_ == 14:
        return "riding8"
    if no_ == 15:
        return "truck1"
    return 0


def draw_result (image, bbox_pred, bbox_gtruth, label_pred):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    tran_img = torch.from_numpy(np.transpose(image.numpy(), (1, 2, 0)))
    ax.imshow(tran_img)
    [r_pred, c_pred, w_pred, h_pred] = [x.item() for x in bbox_pred]
    ax.add_patch(mpatches.Rectangle(
        (r_pred, c_pred), w_pred, h_pred, fill=False, edgecolor="red", linewidth=1))

    [r_gt, c_gt, w_gt, h_gt] = [x.item() for x in bbox_gtruth]
    ax.add_patch(mpatches.Rectangle(
        (r_gt, c_gt), w_gt, h_gt, fill=False, edgecolor="green", linewidth=1))

    iou = utils.get_IOU(bbox_pred, bbox_gtruth)
    plt.text(r_pred + w_pred, c_pred - h_pred, get_label_by_number(label_pred) + '\n' + str(round(iou, 2)))
    plt.savefig(root_path + "test_results\\" + str(datetime.now())[-6:] + ".png")
    plt.close()


all_clf_scores = []
all_bbox_pred = []
for m in range(len(test_data)):
    test_inputs = test_data[m: m+1]
    feature_maps = our_net.forward_feature(test_inputs)

    for i in range(len(test_rg_projs_4_imgs[m])):
        region_projs = [test_rg_projs_4_imgs[m][i]]  # the i-th region for the m-th image
        clf_scores, bbox_pred = our_net.forward_output(feature_maps, region_projs)
        all_clf_scores.append(clf_scores)
        all_bbox_pred.append(bbox_pred)

    result = [0, 0, ()]
    cls_num = len(all_clf_scores[0][0])
    for k in range(len(all_clf_scores)):
        max_p = max(all_clf_scores[k][1:])  # ignore the 0 class (background)
        if max_p > result[0]:
            u = [z for z in range(1, cls_num) if all_clf_scores[k][0][z] == max_p][0]
            results = [max_p, u, all_bbox_pred[k][u]]
    draw_result(test_data[m], results[2], test_label[m][1:], test_label[m][0])
