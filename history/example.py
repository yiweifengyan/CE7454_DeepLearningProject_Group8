import numpy as np
from shutil import copyfile, rmtree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import torch
import readimg_new
import os
from datetime import datetime

def get_region_proposal (img):
    # perform selective search0-
    img_lbl, regions = selectivesearch.selective_search(img, scale=450, sigma=0.6, min_size=5)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue

        if r['size'] < 100:
            continue

        if r['rect'][2] > 360 * 3 / 4 or r['rect'][3] > 640 * 3 / 4:
            continue

        if r['rect'][2] < 3 or r['rect'][3] < 3:
            continue

        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    #plt.show()
    plt.savefig("D:/CE7454_DeepLearningProject_Group8/region_proposals/" + str(datetime.now())[-6:] + '.png')
    plt.close()


if __name__ == "__main__":
    rmtree("D:/CE7454_DeepLearningProject_Group8/region_proposals/")
    os.makedirs("D:/CE7454_DeepLearningProject_Group8/region_proposals/")
    train_data, train_label = readimg_new.read_data(["test_data", "test_label"])
    for tmp in train_data:
        img = torch.from_numpy(np.transpose(tmp.numpy(), (1, 2, 0)))
        get_region_proposal(img)
