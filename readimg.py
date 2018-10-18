from PIL import Image
import os
imdir=os.getcwd()
print(imdir)
imgname=imdir+'\\train_data\\000001.jpg'
print(imgname)
im=Image.open(imgname)
