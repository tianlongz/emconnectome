import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as keras
from keras import losses
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from data import *
# import matplotlib.backends.backend_tkagg
# import matplotlib.pyplot as plt

class myUnet(object):
	def __init__(self, img_rows = 512, img_cols = 512):
		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):
		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		# return imgs_train, imgs_mask_train
		return imgs_train, imgs_mask_train, imgs_test
# loss function
	def dc_loss(self, y_true, y_pred):
		for x,y in y_true.index(9):
			y_pred = self.set_value(matrix, x, y, 9)
		x_ent = K.binary_crossentropy(y_true, y_pred)
		return x_ent

	def set_value(matrix, x, y, val):
	    # 得到张量的宽和高，即第一维和第二维的Size
	    w = int(matrix.get_shape()[0])
	    h = int(matrix.get_shape()[1])
	    # 构造一个只有目标位置有值的稀疏矩阵，其值为目标值于原始值的差
	    val_diff = val - matrix[x][y]
	    diff_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[x, y], values=[val_diff], dense_shape=[w, h]))
	    # 用 Variable.assign_add 将两个矩阵相加
	    return matrix.assign_add(diff_matrix)

# sample weight
	def create_sample_weight(self, imgs_train):
		w = (5/1000)*np.ones(imgs_train.shape[0])
		w[0:5] = np.ones(5)
		return w

# fine tuning 
	def train(self):
		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		w = self.create_sample_weight(imgs_train)
		print("loading data done")
		model = load_model('my_unet.hdf5')
		print("got unet")
		# model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
		# history = LossHistory()
		model.compile(optimizer = Adam(lr = 1e-4), loss = self.dc_loss, metrics = ['accuracy'])
		model_checkpoint = ModelCheckpoint('fine_tuned_unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=2, verbose=1, sample_weight = w, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
		#绘制acc-loss曲线
		# history.loss_plot('epoch')
		print('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('../results/imgs_mask_test.npy', imgs_mask_test)
	
	def test(self):
		#模型评估
		score = model.evaluate(X_test, Y_test, verbose=0)
		print('Test score:', score[0])
		print('Test accuracy:', score[1])

	def save_img(self):
		print("array to image")
		imgs = np.load('../results/imgs_mask_test.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("../results/%d.jpg"%(i))

	def plot_model(self):
		plot_model(model, to_file='model.png')

# class LossHistory(keras.callbacks.Callback):
# 	def on_train_begin(self, logs={}):
# 	    self.losses = {'batch':[], 'epoch':[]}
# 	    self.accuracy = {'batch':[], 'epoch':[]}
# 	    self.val_loss = {'batch':[], 'epoch':[]}
# 	    self.val_acc = {'batch':[], 'epoch':[]}

# 	def on_batch_end(self, batch, logs={}):
# 	    self.losses['batch'].append(logs.get('loss'))
# 	    self.accuracy['batch'].append(logs.get('acc'))
# 	    self.val_loss['batch'].append(logs.get('val_loss'))
# 	    self.val_acc['batch'].append(logs.get('val_acc'))

# 	def on_epoch_end(self, batch, logs={}):
# 	    self.losses['epoch'].append(logs.get('loss'))
# 	    self.accuracy['epoch'].append(logs.get('acc'))
# 	    self.val_loss['epoch'].append(logs.get('val_loss'))
# 	    self.val_acc['epoch'].append(logs.get('val_acc'))

# 	def loss_plot(self, loss_type):
# 	    iters = range(len(self.losses[loss_type]))
# 	    plt.figure()
# 	    # acc
# 	    plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
# 	    # loss
# 	    plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# 	    if loss_type == 'epoch':
# 	        # val_acc
# 	        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
# 	        # val_loss
# 	        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
# 	    plt.grid(True)
# 	    plt.xlabel(loss_type)
# 	    plt.ylabel('acc-loss')
# 	    plt.legend(loc="upper right")
# 	    plt.show()


if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
	myunet.save_img()
	# myunet.plot_model()