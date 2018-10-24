import numpy as np
import torch.nn.functional as F
import utils

def inference (net, image):
    pixel_num = np.prod(image.size())
    scores = net(image.view(1, pixel_num))
    probs = F.softmax(scores, dim=1)

    utils.show_prob_drones(probs, image.int())