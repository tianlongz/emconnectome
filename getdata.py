import requests
import os
import math
from PIL import Image

class getdata(object):
	"""
	setting up the url names and parameters
	"""
	def __init__(
			self, 
			img_url = 'http://storage1.connectomes.utah.edu/', 
			rpc = 'RPC1/', 
			section_number = 'TEM/0703/' , 
			tile_path = 'TEM/Leveled/Tileset/', 
			downsample_level = 4, 
			read_img_type = '.png', 
			save_img_type = '.tif',
			train_img_path = '../train/'):
		self.img_url = img_url
		self.rpc = rpc
		self.section_number = section_number
		self.tile_path = tile_path
		self.downsample_number = downsample_level
		self.downsample_level = '00' + str(downsample_level)
		self.read_img_type = read_img_type
		self.save_img_type = save_img_type
		self.train_img_path = train_img_path


	def getimg(self, max_x = 16620, max_y = 18912, size_number = 512):
		# combine url names
		url = self.img_url + self.rpc + self.section_number + self.tile_path + self.downsample_level + '/Leveled_'
		count = 0
		# get the maximun values of the x and y tiles
		tile_x = math.floor(max_x/(size_number*self.downsample_number))
		tile_y = math.floor(max_y/(size_number*self.downsample_number))
		# start pulling .png images and transfer to .tif images
		for x in range(1,tile_x):
			for y in range(1,tile_y):
				# since the url requires 3-digit number, so there is a need to classify the x and y into different categories
				# classify different x and y values for those who are larger than 9 and those who are less than 10
				if x > 9 or y > 9:
					# request the image, open and save them
					# image name is based on the x,y coordinates, just like the url
					img = requests.get(url + 'X0'+str(x)+'_Y0'+str(y)+ self.read_img_type)
					f = open('X0'+str(x)+'Y0'+str(y)+self.read_img_type,'ab')
					f.write(img.content)
					f.close()
					# change the .png images into .tif images and save them
					# image name is based on the count number since the data.py requires the .tif images have integer names starting from 0
					im = Image.open('X0'+str(x)+'Y0'+str(y)+self.read_img_type)
					im = im.convert('RGB')
					im.save(self.train_img_path + str(count)+self.save_img_type)
					
					count = count + 1
				elif x > 9 and y < 10:
					img = requests.get(url + 'X0'+str(x)+'_Y00'+str(y)+ self.read_img_type)
					f = open('X0'+str(x)+'Y00'+str(y)+self.read_img_type,'ab')
					f.write(img.content)
					f.close()
					im = Image.open('X0'+str(x)+'Y00'+str(y)+self.read_img_type)
					im = im.convert('RGB')
					im.save(self.train_img_path + str(count)+self.save_img_type)
					count = count + 1
				elif x < 10 and y > 9:
					img = requests.get(url + 'X00'+str(x)+'_Y0'+str(y)+ self.read_img_type)
					f = open('X00'+str(x)+'Y0'+str(y)+self.read_img_type,'ab')
					f.write(img.content)
					f.close()
					im = Image.open('X00'+str(x)+'Y0'+str(y)+self.read_img_type)
					im = im.convert('RGB')
					im.save(self.train_img_path + str(count)+self.save_img_type)
					count = count + 1
				else:
					img = requests.get(url + 'X00'+str(x)+'_Y00'+str(y)+ self.read_img_type)
					f = open('X00'+str(x)+'Y00'+str(y)+self.read_img_type,'ab')
					f.write(img.content)
					f.close()
					im = Image.open('X00'+str(x)+'Y00'+str(y)+self.read_img_type)
					im = im.convert('RGB')
					im.save(self.train_img_path + str(count)+self.save_img_type)
					count = count + 1
				

if __name__ == '__main__':
	imgdata = getdata()
	print('start pulling images from url')
	imgdata.getimg(max_x = 16620, max_y = 18912, size_number = 512)
	print('download training data done')


