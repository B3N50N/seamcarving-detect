import os
import sys
#disable GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="5"
# os.environ["TF_CPP_MIN_LOG_LEVEL"]= '3'
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
#from tensorflow import float32
from tensorflow import float32
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as mppyplot
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
test_sample = 50

fc_2_node =40000 # 15000

restore_step =0
P0 = 0
KEEP_PROB_SET = 0.8

image_X =0
image_Y =0 

def read_y(filename):
	global subimage_size
	global result_y1
	global result_y2
	global result_y3
	global image_X
	global image_Y
	
	data_list = []
	output_list = [[0 for i in range(subimage_size)] for j in range(subimage_size)]
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
	image_X = width 
	image_Y = height 

	output_list = np.array(output_list)
	data_list =np.array(data_list)
	output_list[0:width, 0:height] = data_list
	output_list[output_list >1] = 0
	output_list[output_list == 0] = -1 ## use hyper tanh	
	if label_scale >1 : ## down-sample
		output_list = downscale_local_mean(output_list, (label_scale,label_scale))
		# output_list[output_list > 0] = 1
	
	return output_list
	
def read_x(filename):
	global subimage_size
	global result_x1
	global result_x2
	global result_x3
	
	image_decoded =mpimg.imread(filename)
	width = len(image_decoded)
	height = len(image_decoded[0])
	num_channels = len(image_decoded[0][0])
	subimage = [[[0 for i in range(3)] for j in range(subimage_size)]for k in range(subimage_size)]#set subimage size
	subimage =np.array(subimage)
	image_decoded =np.array(image_decoded)
	subimage[0:width, 0:height] = image_decoded	
		
	return subimage
	
def read_from_disk(filename,filename2) :
	x = read_x(filename)
	y = read_y(filename2)
	return x , y
	
def weight_variable(shape,names):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name=names)

def weight_variable_xavier(shape,names):
    
    return tf.Variable(tf.get_variable(name =names, shape=shape,initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32)),name=names)

def bias_variable(shape,names):
    initial = tf.constant(0.5,dtype=tf.float32, shape=shape)
    return tf.Variable(initial,name=names)

def bias_variable_xavier(shape,names):
    return tf.get_variable(name =names, shape=shape,initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),name=names)

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
	xs = tf.placeholder(tf.float32, [batch_size,subimage_size, subimage_size,3])
	 

## prepare sample ##
with tf.device('/device:GPU:1'):#'/device:GPU:0'
	xs_tensor = image_standar(xs)
	x_image = tf.reshape(xs_tensor, [-1, 512, 512, 3])
	

## conv1 layer ##
with tf.device('/device:GPU:0'):
	W_conv1 = weight_variable_xavier([5,5, 3,32],"W_conv1") # patch 5x5, in size 1, out size 32
	b_conv1 = bias_variable([32],"b_conv1")
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1,"h_conv1") + b_conv1, name="h_conv1") # output size 512x512x32
	h_pool1 = avg_pool_2x2(h_conv1,"h_pool1") # output size 256x256x32
	
## conv2 layer ##
with tf.device('/device:GPU:0'):
	W_conv2 = weight_variable_xavier([5,5, 32, 64],"W_conv2") # patch 5x5, in size 32, out size 64
	b_conv2 = bias_variable([64],"b_conv2")
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,"h_conv2") + b_conv2, name="h_conv2") # output size 256x256x64
	h_pool2 = avg_pool_2x2(h_conv2,"h_pool2") # output size 128x128x64
	
## conv3 layer ##
with tf.device('/device:GPU:0'):
	W_conv3 = weight_variable_xavier([5,5, 64, 128],"W_conv3") # patch 5x5, in size 64, out size 128
	b_conv3 = bias_variable([128],"b_conv3")
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3,"h_conv3") + b_conv3, name="h_conv3") # output size 128x128x128
	h_pool3 = avg_pool_2x2(h_conv3,"h_pool3") # output size 64x64x128
	
## conv4 layer ##
with tf.device('/device:GPU:0'):
	W_conv4 = weight_variable_xavier([5,5, 128, 256],"W_conv4") # patch 5x5, in size 128, out size 256
	b_conv4 = bias_variable([256],"b_conv4")
	h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4,"h_conv4") + b_conv4, name="h_conv4") # output size 64*64*256
	h_pool4 = avg_pool_2x2(h_conv4,"h_pool4") # output size 32x32x256
	
## conv5 layer ##
with tf.device('/device:GPU:0'):
	W_conv5 = weight_variable_xavier([5,5, 256, 512],"W_conv5") # patch 5x5, in size 256, out size 512
	b_conv5 = bias_variable([512],"b_conv5")
	h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5,"h_conv5") + b_conv5, name="h_conv5") # output size 32*32*512
	h_pool5 = avg_pool_2x2(h_conv5,"h_pool5") # output size 16x16x512
	
## conv6 layer ##
with tf.device('/device:GPU:0'):
	W_conv6 = weight_variable_xavier([5,5, 512, 1024],"W_conv6") # patch 5x5, in size 256, out size 1024
	b_conv6 = bias_variable([1024],"b_conv6")
	h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6,"h_conv6") + b_conv6, name="h_conv6") # output size 16*16*1024
	h_pool6 = avg_pool_2x2(h_conv6,"h_pool6") # output size 8x8x1024

	
## conv7 layer ##
with tf.device('/device:GPU:0'):
	W_conv7 = weight_variable_xavier([5,5, 1024, 2048],"W_conv7") # patch 5x5, in size 256, out size 1024
	b_conv7 = bias_variable([2048],"b_conv7")
	h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7,"h_conv7") + b_conv7, name="h_conv7") # output size 8*8*2048
	h_pool7 = avg_pool_2x2(h_conv7,"h_pool7") # output size 4x4x2048

## conv8 layer ##	
with tf.device('/device:GPU:0'):
	W_conv8 = weight_variable_xavier([4,4, 2048, 4096],"W_conv8") # patch 5x5, in size 256, out size 1024
	b_conv8 = bias_variable([4096],"b_conv8")
	h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8,"h_conv8") + b_conv8, name="h_conv8") # output size 4*4*4096
	h_pool8 = avg_pool_2x2(h_conv8,"h_pool8") # output size 2*2*4096
	
	
## fc1 layer ##
with tf.device('/device:GPU:0'):
	W_fc1 = weight_variable([2*2*4096, fc_2_node],"W_fc1")#65536,30000
	b_fc1 = bias_variable([fc_2_node],"b_fc1")
	h_pool8_flat = tf.reshape(h_pool8, [-1, 2*2*4096])
	h_fc1 = tf.nn.tanh(tf.matmul(h_pool8_flat, W_fc1) + b_fc1,name="h_fc1")#tf.nn.relu
	h_fc1_drop = tf.nn.dropout(h_fc1, KEEP_PROB_SET, name="h_fc1_drop")
	h_fc1_drop_1 = tf.where(tf.is_nan(h_fc1_drop), tf.constant(min_clip,dtype=tf.float32, shape=h_fc1_drop.shape), h_fc1_drop)
	
##down sampleing fc4 layer##
with tf.device('/device:GPU:0'):
	W_fc4 = weight_variable([fc_2_node,((subimage_size//label_scale)//2)],"W_fc4")
	b_fc4 = bias_variable([((subimage_size//label_scale)//2)],"b_fc4")
	h_fc4 = tf.nn.tanh(tf.matmul(h_fc1_drop_1, W_fc4) + b_fc4, name="h_fc4")#tf.nn.relu h_fc3_1
	h_fc4_1 = tf.where(tf.is_nan(h_fc4), tf.constant(min_clip,dtype=tf.float32, shape=h_fc4.shape), h_fc4)
	
## fc6 layer method 2##
with tf.device('/device:GPU:0'):
	W_fc6 = weight_variable([((subimage_size//label_scale)//2), (subimage_size//label_scale)*(subimage_size//label_scale)],"W_fc6")#262,144
	b_fc6 = bias_variable([(subimage_size//label_scale)*(subimage_size//label_scale)],"b_fc6")
	prediction = tf.nn.tanh(tf.matmul(h_fc4_1, W_fc6) + b_fc6, name="prediction") #tf.nn.sigmoid
	prediction_1 = tf.where(tf.is_nan(prediction), tf.constant(0.0,dtype=tf.float32, shape=prediction.shape), prediction)
	prediction_2 =tf.reshape(prediction_1, [(subimage_size//label_scale),(subimage_size//label_scale)])

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
		saver.restore(sess, "net10/save_net.ckpt")
		print("Model restored.")
	except:
		print("no saved network found")

#train control
restore_net()
filename = "sampleimage/Pre_QF/QF_100/seamcarving10_l/ucid00020.jpg" # 54
filename2 ="sampleimage/Pre_QF/QF_100/seamcarving10_l_txt/ucid00020.txt"
# filename = input("input file name：")
# filename2 = input("input label file name：")
p0 = read_from_disk(filename,filename2)
bx,by = p0
batch_x = []
batch_x.append(bx)
predict = sess.run(prediction_2, feed_dict = {xs: batch_x})
bx = np.reshape(np.array(predict),[(subimage_size//label_scale),(subimage_size//label_scale)])

# pos proccesss prediction
output_list = [[-1 for i in range(subimage_size//label_scale)] for j in range(subimage_size//label_scale)]
output_list = np.array(output_list)
for i in range(0,image_X//label_scale) :
	for j in range(0,image_Y//label_scale) :
		output_list[i,j] = bx[i,j]
bx = output_list

mppyplot.figure()
mppyplot.title('predict')
mppyplot.imshow(bx)
mppyplot.figure()
mppyplot.title('label')
mppyplot.imshow(by)
mppyplot.show()