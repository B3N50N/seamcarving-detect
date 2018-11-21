import os
import sys
#disable GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="5"
os.environ["TF_CPP_MIN_LOG_LEVEL"]= '3'
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
#from tensorflow import float32
from tensorflow import float16
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import time
# import threading as mp
from threading import Thread, Event
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean




#float16:
min_clip = 1e-4
max_clip =65500.0

subimage_size =512
batch_size = 1
batch_size_old =batch_size 
reapeat_time = 1
label_scale =8 # 2

fc_2_node =6500 # 15000
conv_base =10 #32

restore_step =0
P0 = 0
KEEP_PROB_SET = 1.0

image_X =0
image_Y =0 
image_original =0

def read_y(filename):
	global subimage_size
	global result_y1
	global result_y2
	global result_y3
	
	data_list = []
	output_list = [[0 for i in range(subimage_size)] for j in range(subimage_size)]
	output = [[[0 for i in range(subimage_size)] for j in range(subimage_size)] for k in range(2)]
	fileptr_in = open(filename, 'r')
	while True:
		filein_line = fileptr_in.readline()
		if not filein_line: break
		data_list_sec = []	
		for i in range(len(filein_line)-1):
			data_list_sec.append(filein_line[i])
		data_list.append(data_list_sec)
	
	fileptr_in.close()
	
	# data_list = np.loadtxt(filename, dtype='int', delimiter=" ", ndmin=1)
	
	width = len(data_list)
	height = len(data_list[0])

	output_list = np.array(output_list)
	data_list =np.array(data_list)
	output_list[0:width, 0:height] = data_list
	output_list[output_list >1] = 0
	output_list[output_list == 0] = -1 ## use hyper tanh	
	if label_scale >1 : ## down-sample
		output_list = downscale_local_mean(output_list, (label_scale,label_scale))
		# output_list[output_list > 0] = 1
		output_list =((np.sqrt((output_list+1)*50)*10)/50)-1
	
	output[0]= output_list
	output[1]= output_list*-1
	
	# return output_list
	return output
	
def read_x(filename):
	global subimage_size
	global result_x1
	global result_x2
	global result_x3
	global image_original
	
	image_decoded =mpimg.imread(filename)
	width = len(image_decoded)
	height = len(image_decoded[0])
	num_channels = len(image_decoded[0][0])
	subimage = [[[0 for i in range(3)] for j in range(subimage_size)]for k in range(subimage_size)]#set subimage size
	subimage =np.array(subimage)
	image_decoded =np.array(image_decoded)
	subimage[0:width, 0:height] = image_decoded	
	image_original =subimage 
		
	return subimage
	
def read_from_disk(filename,filename2) :
	x = read_x(filename)
	y = read_y(filename2)
	return x , y
	
def weight_variable(shape,names):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float16)
    return tf.Variable(initial,name=names)

def weight_variable_xavier(shape,names):
    
    return tf.Variable(tf.cast(tf.get_variable(name =names, shape=shape,initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16)),tf.float16),name=names)

def bias_variable(shape,names):
    initial = tf.constant(0.5,dtype=tf.float16, shape=shape)
    return tf.Variable(initial,name=names)

def bias_variable_xavier(shape,names):
    return tf.cast(tf.get_variable(name =names, shape=shape,initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float16)),tf.float16,name=names)

def conv3d(x, W, names):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME', name=names)

def conv2d(x, W, names):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=names)

def max_pool_2x2x1(x,names):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool3d(x, ksize=[1,2,2,1,1], strides=[1,2,2,1,1], padding='SAME',name=names)
	
def max_pool_2x2(x,names):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME',name=names)

def avg_pool_2x2x1(x,names):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.avg_pool(x, ksize=[1,2,2,1,1], strides=[1,2,2,1,1], padding='SAME',name=names)

def avg_pool_2x2(x,names):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.avg_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME',name=names)

def image_standar(xs):## write thread
	xs_tensor= [[[[0 for i in range(3)] for j in range(subimage_size)]for k in range(subimage_size)]for l in range(batch_size)]
	for i in range(0,batch_size):
		xs_tensor[i] = tf.image.per_image_standardization(tf.convert_to_tensor(xs[i]))
	return xs_tensor


# define placeholder for inputs to network
with tf.device('/cpu:0'):
	xs = tf.placeholder(tf.float16, [batch_size,subimage_size, subimage_size,3])
	ys = tf.placeholder(tf.float16, [batch_size,2,subimage_size//label_scale, subimage_size//label_scale])

## prepare sample ##
with tf.device('/device:GPU:1'):#'/device:GPU:0'
	xs_tensor = image_standar(xs)
	x_image = tf.cast(tf.reshape(xs_tensor, [-1, 512, 512, 3]),tf.float16)
	

## conv1 layer ##
with tf.device('/device:GPU:0'):
	W_conv1 = weight_variable_xavier([10,10, 3,1*conv_base],"W_conv1") # patch 5x5, in size 1, out size 32
	b_conv1 = bias_variable([1*conv_base],"b_conv1")
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1,"h_conv1") + b_conv1, name="h_conv1") # output size 512x512x32
	h_pool1 = avg_pool_2x2(h_conv1,"h_pool1") # output size 256x256x32
	# h_pool1_LRN = tf.nn.lrn(h_pool1,name="h_pool1_LRN")# output size 256x256x32
	
## conv2 layer ##
with tf.device('/device:GPU:0'):
	W_conv2 = weight_variable_xavier([7,7, 1*conv_base, 2*conv_base],"W_conv2") # patch 5x5, in size 32, out size 64
	b_conv2 = bias_variable([2*conv_base],"b_conv2")
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,"h_conv2") + b_conv2, name="h_conv2") # output size 256x256x64
	h_pool2 = avg_pool_2x2(h_conv2,"h_pool2") # output size 128x128x64
	# h_pool2_LRN = tf.nn.lrn(h_pool2,name="h_pool2_LRN")# output size 128x128x64
	
## conv3 layer ##
with tf.device('/device:GPU:0'):
	W_conv3 = weight_variable_xavier([5,5, 2*conv_base, 4*conv_base],"W_conv3") # patch 5x5, in size 64, out size 128
	b_conv3 = bias_variable([4*conv_base],"b_conv3")
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3,"h_conv3") + b_conv3, name="h_conv3") # output size 128x128x128
	h_pool3 = avg_pool_2x2(h_conv3,"h_pool3") # output size 64x64x128
	# h_pool3_LRN = tf.nn.lrn(h_pool3,name="h_pool3_LRN")# output size 64x64x128
	
	
## fc1 layer ##
with tf.device('/device:GPU:0'):
	W_fc1 = weight_variable([64*64*4*conv_base, fc_2_node],"W_fc1")#65536,3000
	b_fc1 = bias_variable([fc_2_node],"b_fc1")
	h_pool_flat = tf.reshape(h_pool3, [-1, 64*64*4*conv_base])
	h_fc1 = tf.nn.tanh(tf.matmul(h_pool_flat, W_fc1) + b_fc1,name="h_fc1")#tf.nn.relu
	h_fc1_drop = tf.cast(tf.nn.dropout(tf.cast(h_fc1,tf.float32), KEEP_PROB_SET),tf.float16, name="h_fc1_drop")
	h_fc1_drop_1 = tf.where(tf.is_nan(h_fc1_drop), tf.constant(min_clip,dtype=tf.float16, shape=h_fc1_drop.shape), h_fc1_drop)
	
##down sampleing fc4 layer##
with tf.device('/device:GPU:0'):
	W_fc4 = weight_variable([fc_2_node,((subimage_size//label_scale)//2)],"W_fc4")
	b_fc4 = bias_variable([((subimage_size//label_scale)//2)],"b_fc4")
	h_fc4 = tf.nn.tanh(tf.matmul(h_fc1_drop_1, W_fc4) + b_fc4, name="h_fc4")#tf.nn.relu h_fc3_1
	h_fc4_1 = tf.where(tf.is_nan(h_fc4), tf.constant(min_clip,dtype=tf.float16, shape=h_fc4.shape), h_fc4)
	
## fc6 layer method 2##
with tf.device('/device:GPU:0'):
	W_fc6 = weight_variable([((subimage_size//label_scale)//2), 2*(subimage_size//label_scale)*(subimage_size//label_scale)],"W_fc6")#262,144
	b_fc6 = bias_variable([2*(subimage_size//label_scale)*(subimage_size//label_scale)],"b_fc6")
	prediction = tf.nn.tanh(tf.matmul(h_fc4_1, W_fc6) + b_fc6, name="prediction") #tf.nn.sigmoid
	prediction_1 = tf.where(tf.is_nan(prediction), tf.constant(0.0,dtype=tf.float16, shape=prediction.shape), prediction)
	prediction_2 = tf.reshape(prediction_1,[(batch_size),2,(subimage_size//label_scale),(subimage_size//label_scale)])

with tf.device('/device:GPU:0'):
# pre progress y
	# y_reshape = tf.reshape(ys, [-1, (subimage_size//label_scale)*(subimage_size//label_scale)])
	y_reshape = ys

#the error between prediction and real data
	loss = tf.losses.mean_squared_error(y_reshape,prediction_2)

##setting
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.allow_soft_placement=True
# config.log_device_placement=True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
print("start initialization !!")
sess.run(init)
saver = tf.train.Saver()
sess.graph.finalize()


def restore_net():
	global restore_step
	try:
		print("Model restoring.")
		# saver.restore(sess, "net_backup/save_net.ckpt")
		saver.restore(sess, "net13/save_net.ckpt")
		print("Model restored.")
	except:
		print("no saved network found")

#train control
restore_net()
filename = "sampleimage/Pre_QF/QF_100/seamcarving20_l/ucid00020.jpg" # 54 20
filename2 ="sampleimage/Pre_QF/QF_100/seamcarving20_l_txt/ucid00020.txt"
p = "no"
# filename = input("input file name：")
# filename2 = input("input label file name：")
# p = input("show every image in a fig (yes/no)：")
print("proccessing")
p0 = read_from_disk(filename,filename2)
bx,by = p0
batch_x = []
batch_y =[]
batch_x.append(bx)
batch_y.append(by)
predict = sess.run(prediction_2, feed_dict = {xs: batch_x})
losses = sess.run(loss, feed_dict = {xs: batch_x , ys: batch_y})
# bx = np.reshape(np.array(predict),[(subimage_size//label_scale),(subimage_size//label_scale)])
bx = predict[0]
by = by[0]

# pos proccesss prediction
output_list = [[-1 for i in range(subimage_size//label_scale)] for j in range(subimage_size//label_scale)]
output_list = np.array(output_list)
for i in range(0,image_X//label_scale) :
	for j in range(0,image_Y//label_scale) :
		output_list[i,j] = bx[i,j]
bx = output_list
dff = by-bx
dff_2 =np.absolute(dff)/2

MSE = ("MSE = "+str(losses))

for i in range(0,image_Y//label_scale) :
		bx[image_X//label_scale-1,i] =1
		by[image_X//label_scale-1,i] =1
		dff[image_X//label_scale-1,i] =1
		dff_2[image_X//label_scale-1,i] =1
		
for i in range(0,image_X//label_scale) :
		bx[i,image_Y//label_scale-1] =1
		by[i,image_Y//label_scale-1] =1
		dff[i,image_Y//label_scale-1] =1
		dff_2[i,image_Y//label_scale-1] =1


if p =="yes" or p == "y" :
	plt.figure()
	plt.title("predict")
	plt.imshow(bx)
	plt.figure()
	plt.title("label")
	plt.imshow(by)
	plt.figure()
	plt.title("dff = label - predict")
	plt.imshow(dff)
	plt.figure()
	plt.title("dff-2 = |label - predict|/2")
	plt.imshow(dff_2)
	plt.show()

plt.figure(figsize=(8, 6), dpi=160)
plt.suptitle(MSE)
plt.subplot(2,2,1)
plt.title("predict")
plt.imshow(bx)
plt.colorbar()
plt.subplot(2,2,2)
plt.title("label")
plt.imshow(by)
plt.colorbar()
plt.subplot(2,2,3)
# plt.title("dff = label - predict")
# plt.imshow(dff)
plt.title("original image")
plt.imshow(image_original)
plt.colorbar()
plt.subplot(2,2,4)
plt.title("dff-2 = |label - predict|/2")
plt.imshow(dff_2)
plt.colorbar()
plt.tight_layout()

plt.savefig('model10.png')
plt.show()



