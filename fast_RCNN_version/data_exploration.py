import os
import torch
import torchvision
from PIL import Image
from scipy.stats import entropy
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from xml.dom.minidom import parse

root_path = os.getcwd()
path = root_path + "\\raw\\"
folders = os.listdir(path)

# get bounding box ground truth
def get_box (xml_file):
    xmlname = path + folder + '\\' + xml_file
    xml = parse(xmlname)
    xmlcoll = xml.documentElement

    xmin = int(xmlcoll.getElementsByTagName("xmin")[0].childNodes[0].nodeValue)
    xmax = int(xmlcoll.getElementsByTagName("xmax")[0].childNodes[0].nodeValue)
    ymin = int(xmlcoll.getElementsByTagName("ymin")[0].childNodes[0].nodeValue)
    ymax = int(xmlcoll.getElementsByTagName("ymax")[0].childNodes[0].nodeValue)

    return xmin, xmax, ymin, ymax

fig = plt.figure()
plt.title("Entropy Diff. between bounding boxes of the 0th and the ith image")
plt.xlabel("i")
plt.ylabel("Entropy Diff.")
for folder in folders:
    print(folder)
    files = [f for f in os.listdir(path + folder + "\\") if '.jpg' in f]

    gray_converter = torchvision.transforms.Grayscale(1)
    tensor_converter = torchvision.transforms.ToTensor()
    folder_entrs = []
    for f in files:
        image = Image.open(path + folder + "\\" + f)
        xmin, xmax, ymin, ymax = get_box(f.split('.')[0] + '.xml')

        # convert rgb image to gray-scale image
        image = gray_converter(image)

        # convert to tensor
        imtensor = tensor_converter(image) * 255
        imtensor = imtensor.type(torch.uint8).squeeze(0)

        # extract the box and count image entropy
        boxtensor = imtensor[ymin:(ymax+1), xmin:(xmax+1)]
        boxtensor = boxtensor.contiguous().view(1, -1).squeeze(0)
        counter_dict = Counter(boxtensor.numpy())
        pk = []
        for i in range(0, 256):
            if i in counter_dict.keys():
                pk.append(counter_dict[i])
            else:
                pk.append(0)
        entr = entropy(pk)
        folder_entrs.append(entr)

    entr_diff = [round(abs(folder_entrs[i] - folder_entrs[0]), 4) for i in range(0, len(folder_entrs), 10)]

    plt.plot(np.arange(0, len(folder_entrs), 10), entr_diff, linestyle='-', color=np.random.rand(3, ), label=folder)
    #plt.legend()
    #plt.savefig(root_path + "\\data_exploration\\frame\\" + folder + ".png")
    #plt.close()
plt.show()
