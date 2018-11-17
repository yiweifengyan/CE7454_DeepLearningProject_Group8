import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import torch
import numpy as np
import readimg_new
import utils
import os
import random as rm
from datetime import datetime

segment = "segment-random"
train_rp_num = 32
test_rp_num = 1000
first_threshold = 0.1
second_threshold = 0.5

rm.seed(datetime.now())

def get_relaxed_candidates (rps, ious, label):
    sorted_rps = [list(z[1]) + [label.item()] if z[0] > 0.5 else list(z[1]) + [0] for z in sorted(zip(ious, rps), reverse=True)]
    return sorted_rps[0: train_rp_num]

def get_region_proposal (img):
    # perform selective search0-
    img_lbl, regions1 = selectivesearch.selective_search(img, scale=50, sigma=0.8, min_size=5)
    img_lbl, regions2 = selectivesearch.selective_search(img, scale=200, sigma=0.8, min_size=5)
    img_lbl, regions3 = selectivesearch.selective_search(img, scale=300, sigma=0.8, min_size=5)
    img_lbl, regions4 = selectivesearch.selective_search(img, scale=500, sigma=0.8, min_size=5)
    pool = set()
    for r in regions1 + regions2 + regions3 + regions4:
        # excluding same rectangle (with different segments)
        if r['rect'] in pool:
            continue
        if r['size'] < 100:
            continue
        if r['rect'][2] > 640 * 3 / 4 or r['rect'][3] > 360 * 3 / 4:
            continue
        if r['rect'][2] < 5 or r['rect'][3] < 5:
            continue
        pool.add(r['rect'])
    return pool

flag = "test"

train_data, train_label = readimg_new.read_data([segment + "/" + flag + "_data", segment + "/" + flag + "_label"])
root_path = os.getcwd() + '/'
file_names = [z.split('.')[0] for z in readimg_new.file_name(root_path + segment + "/" + flag + "_data")]
completed = [z.split('.')[0] for z in readimg_new.file_name(root_path + segment + "/region_proposals_train")] + [z.split('.')[0] for z in readimg_new.file_name(root_path + segment + "/region_proposals_test")]

for i in range(len(train_data)):
    if file_names[i] in completed:
        continue
    img = torch.from_numpy(np.transpose(train_data[i].numpy(), (1, 2, 0)))
    pool = list(get_region_proposal(img))
    bbox_gt = train_label[i][1:]
    label = train_label[i][0]
    print(i, ": ", str(len(pool)))
    '''
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in pool:
        ax.add_patch(mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1))
    [x, y, w, h] = bbox_gt
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='green', linewidth=1))
    plt.show()
    '''

    candidates_obj = []
    candidates_backgd = []
    candidates_all = []
    ious_obj = []
    ious_backgd = []
    ious_all = []
    for rp in pool:
        iou = utils.get_IOU(bbox_gt, rp)
        candidates_all.append(rp)
        ious_all.append(iou)
        if iou >= first_threshold:
            if iou > second_threshold:
                candidates_obj.append(rp)
                ious_obj.append(iou)
            else:
                candidates_backgd.append(rp)
                ious_backgd.append(iou)

    if flag == "train":
        if len(ious_obj) < train_rp_num * 0.25 or len(ious_backgd) < train_rp_num * 0.75:
            print(file_names[i], ", obj_num not enough: ", len(ious_obj), ", backgd_num not enough: ", len(ious_backgd))
            candidates = get_relaxed_candidates (candidates_all, ious_all, label)
        else:
            candidates = [list(candidates_obj[l]) + [label.item()] for l in range(int(train_rp_num * 0.25))] + [list(candidates_backgd[l]) + [0] for l in range(int(train_rp_num * 0.75))]

        rm.shuffle(candidates)

        with open(root_path + segment + "/region_proposals_train/" + file_names[i] + ".csv", 'w', newline='') as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(["x_lefttop", "y_lefttop", "width", "height", "label"])
            for j in range(train_rp_num):
                csvwriter.writerow(candidates[j])

    elif flag == "test":
        with open(root_path + segment + "/region_proposals_test/" + file_names[i] + ".csv", 'w', newline='') as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(["x_lefttop", "y_lefttop", "width", "height", "label"])
            for j in range(min(test_rp_num, len(pool))):
                if ious_all[j] > 0.5:
                    csvwriter.writerow(list(pool[j]) + [label.item(), ious_all[j].item()])
                else:
                    csvwriter.writerow(list(pool[j]) + [0, ious_all[j].item()])
