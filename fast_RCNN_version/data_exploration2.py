import os
import matplotlib.pyplot as plt
import numpy as np
from xml.dom.minidom import parse

root_path = os.getcwd()
path = root_path + "\\raw\\"
folders = os.listdir(path)

outputs = []
for folder in folders:
    print(folder)
    files = [f for f in os.listdir(path + folder + "\\") if '.xml' in f]

    max_size = 0
    min_size = 640 * 360
    for f in files:
        xmlname = path + folder + '\\' + f
        xml = parse(xmlname)
        xmlcoll = xml.documentElement

        xmin = int(xmlcoll.getElementsByTagName("xmin")[0].childNodes[0].nodeValue)
        xmax = int(xmlcoll.getElementsByTagName("xmax")[0].childNodes[0].nodeValue)
        ymin = int(xmlcoll.getElementsByTagName("ymin")[0].childNodes[0].nodeValue)
        ymax = int(xmlcoll.getElementsByTagName("ymax")[0].childNodes[0].nodeValue)

        width = xmax - xmin + 1
        height = ymax - ymin + 1
        size = width * height

        if size > max_size:
            max_size = size
        if size < min_size:
            min_size = size
    outputs.append([min_size, max_size])

fig = plt.figure()
plt.title("Bounding box size ranges")
plt.xlabel("the ith class")
plt.ylabel("bounding box size range")
for i in range(0, len(outputs)):
    plt.vlines(i+1, outputs[i][0], outputs[i][1], color='r', linewidth=2)
    plt.text(i+1, outputs[i][1], folders[i], rotation='vertical', fontsize=10)
plt.show()