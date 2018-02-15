import os
import random
from PIL import Image
import numpy as np
from os import listdir
from PIL import ImageFont
from PIL import ImageDraw

command = 'wget http://host.robots.ox.ac.uk/pascal/VOC/download/uiuc.tar.gz'
os.system(command)
os.system('tar -xvf uiuc.tar.gz')
numImages = random.randint(7,20)

os.system('mkdir cropped_images')

pathFiles = os.listdir('UIUC/PNGImages/TestImages')
font_s = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 24)

contador = 1;
while contador < numImages:
	img = Image.open('UIUC/PNGImages/TestImages/test-' + str(contador) +'.png')
	new_img = img.resize ((256,256))
	np.array(new_img)
	np.array(new_img).shape
	new_img.save('cropped_images/test-' + str(contador) + '.png','png')
	f = open('UIUC/Annotations/TestImages/test-' + str(contador) +'.txt')
	lines = f.readlines()
	splitedline = lines[13].split(':')
	annotation = splitedline[1]
	new_img = new_img.convert('RGB')
	draw = ImageDraw.Draw(new_img)
	RED = 255,0,0
	draw.text((10,10),annotation,RED,font=font_s)
	new_img.show()
	if contador == 1:
		nparr=np.array([new_img, annotation])
	else:
		nparr2=np.array([new_img, annotation])
		nparr=np.concatenate((nparr,nparr2))
	print nparr
	contador = contador + 1

