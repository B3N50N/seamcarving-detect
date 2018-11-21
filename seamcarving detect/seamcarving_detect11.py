import os
import sys
#disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="5"
# os.environ["TF_CPP_MIN_LOG_LEVEL"]= '3'
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
#from tensorflow import float32
from tensorflow import float16
import numpy as np
import matplotlib.image as mpimg
#import matplotlib.pyplot as mppyplot
import cv2
import time
from gensim import models
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
batch_size = 28
batch_size_old =28
reapeat_time = 1
label_scale =2

restore_step =0
P0 = 0
substart = time.time()

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.001     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.001  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.
lr = INITIAL_LEARNING_RATE
lr_sum = tf.summary.scalar('learning_rate', lr)
EPOCHS_PRESET = 500
KEEP_PROB_SET = 0.8


STEP_PER_EPOCH_s = (1038*((5*6)+1)*9)*11 #(sample*((src*noise)+normal)*(preqf+posqf+pure))*(smooth+sharp)
STEP_PER_EPOCH_t1 = (1038*((5*6)+1)*9)*11
STEP_PER_EPOCH_t2 = (300*((5*6)+1)*9)*11

STEP_PER_EPOCH_s = (1038*((5*6)+1)*9) #(sample*((src*noise)+normal)*(preqf+posqf+pure))*(smooth+sharp)
STEP_PER_EPOCH_t1 = (1038*((5*6)+1)*9)
STEP_PER_EPOCH_t2 = (300*((5*6)+1)*9)

STEP_PER_EPOCH_s = (100*5*9*7) #(sample*((src*noise)+normal)*(preqf+posqf+pure))*(smooth+sharp)
STEP_PER_EPOCH_t1 = (100*5*9*7)
STEP_PER_EPOCH_t2 = (50*5*9*7)

train_divide = 1038
validate_divide = 1338-train_divide

train_divide = 100
validate_divide = 50-train_divide

#sample image variable
qfm = ["Pre_QF","Pos_QF","pure"]
qf_percent = [10,20,30,50]
quality_factor = [25,50,75,100]
seam_remove = [10,20,30,40,50]
#seam_remove = [11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49]
noise = [5,10,20,30,40,50]
smooth =[3,5,7]
sharp_Radius = [1,1.5]
sharp_Amount = [0.6,1.8]#0.6 1.2 1.8
sharp_Threshold = [0,0.7]# 0 0.4 0.7
qfc = 4
src = 5#36
noisec = 6
smoothc = 3
sharp_Radiusc =  2
sharp_Amountc =  2
sharp_Thresholdc = 2
image_c=100 #1338
fpc = 0

#folder_pool (fp)
folder_pool1 =['_l','_l_noise/seamcarving5_h','_l_noise/seamcarving5_h_sharp_1.5_0.6_0','_l_noise/seamcarving5_h_sharp_1.5_0.6_0.7',
               '_l_noise/seamcarving5_h_sharp_1.5_1.8_0','_l_noise/seamcarving5_h_sharp_1.5_1.8_0','_l_noise/seamcarving5_h_sharp_1.5_1.8_0.7',
			   '_l_noise/seamcarving5_h_sharp_1_0.6_0','_l_noise/seamcarving5_h_sharp_1_0.6_0.7','_l_noise/seamcarving5_h_sharp_1_1.8_0',
			   '_l_noise/seamcarving5_h_sharp_1_1.8_0.7','_l_noise/seamcarving5_h_smooth_3','_l_noise/seamcarving5_h_smooth_5',
			   '_l_noise/seamcarving5_h_smooth_7','_l_noise/seamcarving10_h','_l_noise/seamcarving10_h_sharp_1.5_0.6_0','_l_noise/seamcarving10_h_sharp_1.5_0.6_0.7',
			   '_l_noise/seamcarving10_h_sharp_1.5_1.8_0','_l_noise/seamcarving10_h_sharp_1.5_1.8_0.7','_l_noise/seamcarving10_h_sharp_1_0.6_0',
			   '_l_noise/seamcarving10_h_sharp_1_0.6_0.7','_l_noise/seamcarving10_h_sharp_1_1.8_0','_l_noise/seamcarving10_h_sharp_1_1.8_0.7',
			   '_l_noise/seamcarving10_h_smooth_3','_l_noise/seamcarving10_h_smooth_5','_l_noise/seamcarving10_h_smooth_7','_l_noise/seamcarving20_h',
			   '_l_noise/seamcarving20_h_sharp_1.5_0.6_0','_l_noise/seamcarving20_h_sharp_1.5_0.6_0.7','_l_noise/seamcarving20_h_sharp_1.5_1.8_0',
			   '_l_noise/seamcarving20_h_sharp_1.5_1.8_0.7','_l_noise/seamcarving20_h_sharp_1_0.6_0','_l_noise/seamcarving20_h_sharp_1_0.6_0.7',
			   '_l_noise/seamcarving20_h_sharp_1_1.8_0','_l_noise/seamcarving20_h_sharp_1_1.8_0.7','_l_noise/seamcarving20_h_smooth_3','_l_noise/seamcarving20_h_smooth_5',
			   '_l_noise/seamcarving20_h_smooth_7','_l_noise/seamcarving30_h','_l_noise/seamcarving30_h_sharp_1.5_0.6_0','_l_noise/seamcarving30_h_sharp_1.5_0.6_0.7',
			   '_l_noise/seamcarving30_h_sharp_1.5_1.8_0','_l_noise/seamcarving30_h_sharp_1.5_1.8_0.7','_l_noise/seamcarving30_h_sharp_1_0.6_0',
			   '_l_noise/seamcarving30_h_sharp_1_0.6_0.7','_l_noise/seamcarving30_h_sharp_1_1.8_0','_l_noise/seamcarving30_h_sharp_1_1.8_0.7','_l_noise/seamcarving30_h_smooth_3',
			   '_l_noise/seamcarving30_h_smooth_5','_l_noise/seamcarving30_h_smooth_7','_l_noise/seamcarving40_h','_l_noise/seamcarving40_h_sharp_1.5_0.6_0',
			   '_l_noise/seamcarving40_h_sharp_1.5_0.6_0.7','_l_noise/seamcarving40_h_sharp_1.5_1.8_0','_l_noise/seamcarving40_h_sharp_1.5_1.8_0.7',
			   '_l_noise/seamcarving40_h_sharp_1_0.6_0','_l_noise/seamcarving40_h_sharp_1_0.6_0.7','_l_noise/seamcarving40_h_sharp_1_1.8_0','_l_noise/seamcarving40_h_sharp_1_1.8_0.7',
			   '_l_noise/seamcarving40_h_smooth_3','_l_noise/seamcarving40_h_smooth_5','_l_noise/seamcarving40_h_smooth_7','_l_noise/seamcarving50_h',
			   '_l_noise/seamcarving50_h_sharp_1.5_0.6_0','_l_noise/seamcarving50_h_sharp_1.5_0.6_0.7','_l_noise/seamcarving50_h_sharp_1.5_1.8_0','_l_noise/seamcarving50_h_sharp_1.5_1.8_0.7',
			   '_l_noise/seamcarving50_h_sharp_1_0.6_0','_l_noise/seamcarving50_h_sharp_1_0.6_0.7','_l_noise/seamcarving50_h_sharp_1_1.8_0','_l_noise/seamcarving50_h_sharp_1_1.8_0.7',
			   '_l_noise/seamcarving50_h_smooth_3','_l_noise/seamcarving50_h_smooth_5','_l_noise/seamcarving50_h_smooth_7','_l_sharp_1.5_0.6_0','_l_sharp_1.5_0.6_0.7',
			   '_l_sharp_1.5_1.8_0','_l_sharp_1.5_1.8_0.7','_l_sharp_1_0.6_0','_l_sharp_1_0.6_0.7','_l_sharp_1_1.8_0','_l_sharp_1_1.8_0.7','_l_smooth_3','_l_smooth_5','_l_smooth_7']
folder_pool2 =['_l_txt','_l_noise/seamcarving5_h_txt','_l_noise/seamcarving5_h_txt','_l_noise/seamcarving5_h_txt',
               '_l_noise/seamcarving5_h_txt','_l_noise/seamcarving5_h_txt','_l_noise/seamcarving5_h_txt',
			   '_l_noise/seamcarving5_h_txt','_l_noise/seamcarving5_h_txt','_l_noise/seamcarving5_h_txt',
			   '_l_noise/seamcarving5_h_txt','_l_noise/seamcarving5_h_txt','_l_noise/seamcarving5_h_txt',
			   '_l_noise/seamcarving5_h_txt','_l_noise/seamcarving10_h_txt','_l_noise/seamcarving10_h_txt','_l_noise/seamcarving10_h_txt',
			   '_l_noise/seamcarving10_h_txt','_l_noise/seamcarving10_h_txt','_l_noise/seamcarving10_h_txt',
			   '_l_noise/seamcarving10_h_txt','_l_noise/seamcarving10_h_txt','_l_noise/seamcarving10_h_txt',
			   '_l_noise/seamcarving10_h_txt','_l_noise/seamcarving10_h_txt','_l_noise/seamcarving10_h_txt','_l_noise/seamcarving20_h_txt',
			   '_l_noise/seamcarving20_h_txt','_l_noise/seamcarving20_h_txt','_l_noise/seamcarving20_h_txt',
			   '_l_noise/seamcarving20_h_txt','_l_noise/seamcarving20_h_txt','_l_noise/seamcarving20_h_txt',
			   '_l_noise/seamcarving20_h_txt','_l_noise/seamcarving20_h_txt','_l_noise/seamcarving20_h_txt','_l_noise/seamcarving20_h_txt',
			   '_l_noise/seamcarving20_h_txt','_l_noise/seamcarving30_h_txt','_l_noise/seamcarving30_h_txt','_l_noise/seamcarving30_h_txt',
			   '_l_noise/seamcarving30_h_txt','_l_noise/seamcarving30_h_txt','_l_noise/seamcarving30_h_txt',
			   '_l_noise/seamcarving30_h_txt','_l_noise/seamcarving30_h_txt','_l_noise/seamcarving30_h_txt','_l_noise/seamcarving30_h_txt',
			   '_l_noise/seamcarving30_h_txt','_l_noise/seamcarving30_h_txt','_l_noise/seamcarving40_h_txt','_l_noise/seamcarving40_h_txt',
			   '_l_noise/seamcarving40_h_txt','_l_noise/seamcarving40_h_txt','_l_noise/seamcarving40_h_txt',
			   '_l_noise/seamcarving40_h_txt','_l_noise/seamcarving40_h_txt','_l_noise/seamcarving40_h_txt','_l_noise/seamcarving40_h_txt',
			   '_l_noise/seamcarving40_h_txt','_l_noise/seamcarving40_h_txt','_l_noise/seamcarving40_h_txt','_l_noise/seamcarving50_h_txt',
			   '_l_noise/seamcarving50_h_txt','_l_noise/seamcarving50_h_txt','_l_noise/seamcarving50_h_txt','_l_noise/seamcarving50_h_txt',
			   '_l_noise/seamcarving50_h_txt','_l_noise/seamcarving50_h_txt','_l_noise/seamcarving50_h_txt','_l_noise/seamcarving50_h_txt',
			   '_l_noise/seamcarving50_h_txt','_l_noise/seamcarving50_h_txt','_l_noise/seamcarving50_h_txt','_l_txt','_l_txt',
			   '_l_txt','_l_txt','_l_txt','_l_txt','_l_txt','_l_txt','_l_txt','_l_txt','_l_txt']
		   
## folder slim ver(fp)
folder_pool1 =['_l','_l_noise/seamcarving5_h','_l_noise/seamcarving10_h','_l_noise/seamcarving20_h','_l_noise/seamcarving30_h',
			   '_l_noise/seamcarving40_h','_l_noise/seamcarving50_h']
folder_pool2 =['_l_txt','_l_noise/seamcarving5_h_txt','_l_noise/seamcarving10_h_txt','_l_noise/seamcarving20_h_txt',
			   '_l_noise/seamcarving30_h_txt','_l_noise/seamcarving40_h_txt','_l_noise/seamcarving50_h_txt']
				   
# folder_pool1 =['_l']
# folder_pool2 =['_l_txt']			   

assert len(folder_pool1) ==len(folder_pool2)
fpc = len(folder_pool2)


#read_train variable
qfm_t0 = 0
qfc_t0 = 0
s_start_t0 =1
s_end_t0 = 100+1
image_t0 =s_start_t0
src_t0 =0
fp_t0 =0

#read_test variable
qfm_t1 = 0
qfc_t1 = 0
s_start_t1 =1
s_end_t1 = 100+1
image_t1 =s_start_t1
src_t1 =0
fp_t1 =0

#read_test_2 variable
qfm_t2 = 0
qfc_t2 = 0
s_start_t2 =1039
s_end_t2 = 1088+1
image_t2 =s_start_t2
src_t2 =0
fp_t2 =0


STEP_PER_EPOCH_s = ((s_end_t0-s_start_t0)*src*9*fpc) #(sample*((src*noise)+normal)*(preqf+posqf+pure))*(smooth+sharp)
STEP_PER_EPOCH_t1 = ((s_end_t1-s_start_t1)*src*9*fpc)
STEP_PER_EPOCH_t2 = ((s_end_t2-s_start_t2)*src*9*fpc)


def init_reader():
	#read_train variable
	global qfm_t0
	global qfc_t0
	global s_start_t0
	global image_t0
	global src_t0
	global fp_t0

	#read_test variable
	global qfm_t1
	global qfc_t1
	global s_start_t1
	global image_t1
	global src_t1
	global fp_t1

	#read_test_2 variable
	global qfm_t2
	global qfc_t2
	global s_start_t2
	global image_t2
	global src_t2
	global fp_t2
	#read_train variable
	qfm_t0 = 0
	qfc_t0 = 0
	image_t0 =s_start_t0
	src_t0 =0
	fp_t0 =0

	#read_test variable
	qfm_t1 = 0
	qfc_t1 = 0
	image_t1 =s_start_t1
	src_t1 =0
	fp_t1 =0

	#read_test_2 variable
	qfm_t2 = 0
	qfc_t2 = 0
	image_t2 =s_start_t2
	src_t2 =0
	fp_t2 =0

def filename_maker(qfm_t,qfc_t,src_t,image_t,fp_t):
	global qfm
	global quality_factor
	global seam_remove
	global noise
	global smooth
	global sharp_Radius
	global sharp_Amount
	global sharp_Threshold
	global folder_pool1
	global folder_pool2
	
	#filename = "sampleimage/Pre_QF/QF_100/seamcarving10_l/ucid00001.jpg"
	#filename2 = "sampleimage/Pre_QF/QF_100/seamcarving10_l_txt/ucid00001.txt"
	
	if len(str(image_t)) ==1:
		filename =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool1[fp_t])+"/ucid0000"+str(image_t)+".jpg")
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid0000"+str(image_t)+".txt")
	elif len(str(image_t)) ==2:
		filename =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool1[fp_t])+"/ucid000"+str(image_t)+".jpg")
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid000"+str(image_t)+".txt")
	elif len(str(image_t)) ==3:
		filename =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool1[fp_t])+"/ucid00"+str(image_t)+".jpg")
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid00"+str(image_t)+".txt")
	elif len(str(image_t)) ==4:
		filename =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool1[fp_t])+"/ucid0"+str(image_t)+".jpg")
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid0"+str(image_t)+".txt")
	return filename,filename2

def filename_maker_tif(qfm_t,qfc_t,src_t,image_t,fp_t):
	global qfm
	global quality_factor
	global seam_remove
	global noise
	global smooth
	global sharp_Radius
	global sharp_Amount
	global sharp_Threshold
	global folder_pool1
	global folder_pool2
	
	#filename = "sampleimage/Pre_QF/QF_100/seamcarving10_l/ucid00001.jpg"
	#filename2 = "sampleimage/Pre_QF/QF_100/seamcarving10_l_txt/ucid00001.txt"
	
	if len(str(image_t)) ==1:
		filename =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool1[fp_t])+"/ucid0000"+str(image_t)+".tif")
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid0000"+str(image_t)+".txt")
	elif len(str(image_t)) ==2:
		filename =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool1[fp_t])+"/ucid000"+str(image_t)+".tif")
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid000"+str(image_t)+".txt")
	elif len(str(image_t)) ==3:
		filename =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool1[fp_t])+"/ucid00"+str(image_t)+".tif")
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid00"+str(image_t)+".txt")
	elif len(str(image_t)) ==4:
		filename =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool1[fp_t])+"/ucid0"+str(image_t)+".tif")
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid0"+str(image_t)+".txt")
	return filename,filename2

def downsample(x):
	global subimage_size
	data_list =x
	output_list = [[0 for i in range(int(subimage_size/2))] for j in range(int(subimage_size/2))]
	for w in range(0, subimage_size-1) :
		for h in range(0, subimage_size-1) :
			if (data_list[w][h]==1) or (data_list[w+1][h]==1) or (data_list[w][h+1]==1) or (data_list[w+1][h+1]==1):
				output_list[int(w/2)][int(h/2)] =1
			else:
				output_list[int(w/2)][int(h/2)] =0
			h=h+1
		w=w+1
	#output_list = np.array(output_list, dtype=tf.float32)
	output_list = np.array(output_list)	
	
	return output_list
	
def read_y(filename):
	global subimage_size
	global result_y1
	global result_y2
	global result_y3
	
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

	output_list = np.array(output_list)
	data_list =np.array(data_list)
	output_list[0:width, 0:height] = data_list
	output_list[output_list >1] = 0	
	if label_scale >1 : ## down-sample
		output_list = downscale_local_mean(output_list, (label_scale,label_scale))
		output_list[output_list > 0] = 1
	output_list[output_list == 0] = -1 ## use hyper tanh
	
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
	
def read_from_disk(filename,filename2,return_dict,i) :
	x = read_x(filename)
	y = read_y(filename2)
	return_dict[i] = x , y
	return x , y

def read_train(return_dict_t0,i):
	global qfc
	global src
	global fpc
	global quality_factor
	
	global qfm_t0
	global qfc_t0
	global s_start_t0
	global s_end_t0
	global image_t0
	global src_t0
	global fp_t0
	
	global batch_size
	global reapeat_time
	global subimage_size
	manager = multiprocessing.Manager()
	return_dict = manager.dict()
	jobs = []
	x =[]
	y =[]
	
	for i in range(batch_size) :
		#set file name
		if qfm_t0 < 2:
			if qfc_t0 < qfc-1:
				if src_t0 <src-1:
					if image_t0 < s_end_t0 -1:
						if fp_t0 < fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 =fp_t0 +1
						elif fp_t0 == fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 = 0
							image_t0 =image_t0 +1
					elif image_t0 == s_end_t0 -1:
						if fp_t0 < fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 =fp_t0 +1
						elif fp_t0 == fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 = 0
							image_t0 = s_start_t0
							src_t0 = src_t0 +1
				elif src_t0 ==src-1:
					if image_t0 < s_end_t0 -1:
						if fp_t0 < fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 =fp_t0 +1
						elif fp_t0 == fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 = 0
							image_t0 =image_t0 +1
					elif image_t0 == s_end_t0 -1:
						if fp_t0 < fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 =fp_t0 +1
						elif fp_t0 == fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 = 0
							image_t0 = s_start_t0
							src_t0 =0
							qfc_t0 = qfc_t0 +1
			elif qfc_t0 == qfc-1:
				if src_t0 <src-1:
					if image_t0 < s_end_t0 -1:
						if fp_t0 < fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 =fp_t0 +1
						elif fp_t0 == fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 = 0
							image_t0 =image_t0 +1
					elif image_t0 == s_end_t0 -1:
						if fp_t0 < fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 =fp_t0 +1
						elif fp_t0 == fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 = 0
							image_t0 = s_start_t0
							src_t0 = src_t0 +1
				elif src_t0 ==src-1:
					if image_t0 < s_end_t0 -1:
						if fp_t0 < fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 =fp_t0 +1
						elif fp_t0 == fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 = 0
							image_t0 =image_t0 +1
					elif image_t0 == s_end_t0 -1:
						if fp_t0 < fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 =fp_t0 +1
						elif fp_t0 == fpc-1:
							filename,filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
							fp_t0 = 0
							image_t0 = s_start_t0
							src_t0 =0
							qfc_t0 =0
							qfm_t0 = qfm_t0 +1
		elif qfm_t0 == 2:
			if src_t0 <src-1:
				if image_t0 < s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 =image_t0 +1
				elif image_t0 == s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 = s_start_t0
						src_t0 = src_t0 +1
			elif src_t0 ==src-1:
				if image_t0 < s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 =image_t0 +1
				elif image_t0 == s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 = s_start_t0
						src_t0 =0
						qfm_t0 =0
						print("epoch finish")
		
	
	
		#filename = "sampleimage/Pre_QF/QF_100/seamcarving10_l/ucid00001.jpg"
		#filename2 = "sampleimage/Pre_QF/QF_100/seamcarving10_l_txt/ucid00001.txt"
		
		th1 = Process(target=read_from_disk, args=(filename,filename2,return_dict,i))
		jobs.append(th1)
		th1.start()
	
	for proc in jobs:
		proc.join()
	
	return_temp = return_dict.values()
	for i in range(batch_size) :
		x_temp,y_temp = return_temp[i]
		x.append(x_temp)
		y.append(y_temp)
		
	return_dict_t0[0] = x,y,qfm_t0,qfc_t0,src_t0,image_t0,fp_t0
	return x,y,qfm_t0,qfc_t0,src_t0,image_t0,fp_t0
	
def read_test(x,y):
	global qfc
	global src
	global fpc
	global quality_factor
	
	global qfm_t1
	global qfc_t1
	global s_start_t1
	global s_end_t1
	global image_t1
	global src_t1
	global fp_t1
	
	global batch_size
	global reapeat_time
	global subimage_size
	manager = multiprocessing.Manager()
	return_dict = manager.dict()
	jobs = []
	x =[]
	y =[]
	
	for i in range(batch_size) :
		#set file name
		if qfm_t1 < 2:
			if qfc_t1 < qfc-1:
				if src_t1 <src-1:
					if image_t1 < s_end_t1 -1:
						if fp_t1 < fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 =fp_t1 +1
						elif fp_t1 == fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 = 0
							image_t1 =image_t1 +1
					elif image_t1 == s_end_t1 -1:
						if fp_t1 < fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 =fp_t1 +1
						elif fp_t1 == fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 = 0
							image_t1 = s_start_t1
							src_t1 = src_t1 +1
				elif src_t1 ==src-1:
					if image_t1 < s_end_t1 -1:
						if fp_t1 < fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 =fp_t1 +1
						elif fp_t1 == fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 = 0
							image_t1 =image_t1 +1
					elif image_t1 == s_end_t1 -1:
						if fp_t1 < fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 =fp_t1 +1
						elif fp_t1 == fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 = 0
							image_t1 = s_start_t1
							src_t1 =0
							qfc_t1 = qfc_t1 +1
			elif qfc_t1 == qfc-1:
				if src_t1 <src-1:
					if image_t1 < s_end_t1 -1:
						if fp_t1 < fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 =fp_t1 +1
						elif fp_t1 == fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 = 0
							image_t1 =image_t1 +1
					elif image_t1 == s_end_t1 -1:
						if fp_t1 < fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 =fp_t1 +1
						elif fp_t1 == fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 = 0
							image_t1 = s_start_t1
							src_t1 = src_t1 +1
				elif src_t1 ==src-1:
					if image_t1 < s_end_t1 -1:
						if fp_t1 < fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 =fp_t1 +1
						elif fp_t1 == fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 = 0
							image_t1 =image_t1 +1
					elif image_t1 == s_end_t1 -1:
						if fp_t1 < fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 =fp_t1 +1
						elif fp_t1 == fpc-1:
							filename,filename2 = filename_maker(qfm_t1,str(quality_factor[qfc_t1]),src_t1,image_t1,fp_t1)
							fp_t1 = 0
							image_t1 = s_start_t1
							src_t1 =0
							qfc_t1 =0
							qfm_t1 = qfm_t1 +1
		elif qfm_t1 == 2:
			if src_t1 <src-1:
				if image_t1 < s_end_t1 -1:
					if fp_t1 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t1,"pure",src_t1,image_t1,fp_t1)
						fp_t1 =fp_t1 +1
					elif fp_t1 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t1,"pure",src_t1,image_t1,fp_t1)
						fp_t1 = 0
						image_t1 =image_t1 +1
				elif image_t1 == s_end_t1 -1:
					if fp_t1 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t1,"pure",src_t1,image_t1,fp_t1)
						fp_t1 =fp_t1 +1
					elif fp_t1 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t1,"pure",src_t1,image_t1,fp_t1)
						fp_t1 = 0
						image_t1 = s_start_t1
						src_t1 = src_t1 +1
			elif src_t1 ==src-1:
				if image_t1 < s_end_t1 -1:
					if fp_t1 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t1,"pure",src_t1,image_t1,fp_t1)
						fp_t1 =fp_t1 +1
					elif fp_t1 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t1,"pure",src_t1,image_t1,fp_t1)
						fp_t1 = 0
						image_t1 =image_t1 +1
				elif image_t1 == s_end_t1 -1:
					if fp_t1 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t1,"pure",src_t1,image_t1,fp_t1)
						fp_t1 =fp_t1 +1
					elif fp_t1 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t1,"pure",src_t1,image_t1,fp_t1)
						fp_t1 = 0
						image_t1 = s_start_t1
						src_t1 =0
						qfm_t1 =0
		
	
	
		#filename = "sampleimage/Pre_QF/QF_100/seamcarving10_l/ucid00001.jpg"
		#filename2 = "sampleimage/Pre_QF/QF_100/seamcarving10_l_txt/ucid00001.txt"
		
		th1 = Process(target=read_from_disk, args=(filename,filename2,return_dict,i))
		jobs.append(th1)
		th1.start()
	
	for proc in jobs:
		proc.join()
	
	return_temp = return_dict.values()
	for i in range(batch_size) :
		x_temp,y_temp = return_temp[i]
		x.append(x_temp)
		y.append(y_temp)
		
	# np.testing.assert_array_equal(x,result_x2,err_msg='assert fail')
	# np.testing.assert_array_equal(y,result_y2,err_msg='assert fail')
	
	return x, y
	
def read_test_2(x,y):
	global qfc
	global src
	global fpc
	global quality_factor
	
	global qfm_t2
	global qfc_t2
	global s_start_t2
	global s_end_t2
	global image_t2
	global src_t2
	global fp_t2
	
	global batch_size
	global reapeat_time
	global subimage_size
	manager = multiprocessing.Manager()
	return_dict = manager.dict()
	jobs = []
	x =[]
	y =[]
	
	for i in range(batch_size) :
		#set file name
		if qfm_t2 < 2:
			if qfc_t2 < qfc-1:
				if src_t2 <src-1:
					if image_t2 < s_end_t2 -1:
						if fp_t2 < fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 =fp_t2 +1
						elif fp_t2 == fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 = 0
							image_t2 =image_t2 +1
					elif image_t2 == s_end_t2 -1:
						if fp_t2 < fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 =fp_t2 +1
						elif fp_t2 == fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 = 0
							image_t2 = s_start_t2
							src_t2 = src_t2 +1
				elif src_t2 ==src-1:
					if image_t2 < s_end_t2 -1:
						if fp_t2 < fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 =fp_t2 +1
						elif fp_t2 == fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 = 0
							image_t2 =image_t2 +1
					elif image_t2 == s_end_t2 -1:
						if fp_t2 < fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 =fp_t2 +1
						elif fp_t2 == fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 = 0
							image_t2 = s_start_t2
							src_t2 =0
							qfc_t2 = qfc_t2 +1
			elif qfc_t2 == qfc-1:
				if src_t2 <src-1:
					if image_t2 < s_end_t2 -1:
						if fp_t2 < fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 =fp_t2 +1
						elif fp_t2 == fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 = 0
							image_t2 =image_t2 +1
					elif image_t2 == s_end_t2 -1:
						if fp_t2 < fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 =fp_t2 +1
						elif fp_t2 == fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 = 0
							image_t2 = s_start_t2
							src_t2 = src_t2 +1
				elif src_t2 ==src-1:
					if image_t2 < s_end_t2 -1:
						if fp_t2 < fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 =fp_t2 +1
						elif fp_t2 == fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 = 0
							image_t2 =image_t2 +1
					elif image_t2 == s_end_t2 -1:
						if fp_t2 < fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 =fp_t2 +1
						elif fp_t2 == fpc-1:
							filename,filename2 = filename_maker(qfm_t2,str(quality_factor[qfc_t2]),src_t2,image_t2,fp_t2)
							fp_t2 = 0
							image_t2 = s_start_t2
							src_t2 =0
							qfc_t2 =0
							qfm_t2 = qfm_t2 +1
		elif qfm_t2 == 2:
			if src_t2 <src-1:
				if image_t2 < s_end_t2 -1:
					if fp_t2 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t2,"pure",src_t2,image_t2,fp_t2)
						fp_t2 =fp_t2 +1
					elif fp_t2 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t2,"pure",src_t2,image_t2,fp_t2)
						fp_t2 = 0
						image_t2 =image_t2 +1
				elif image_t2 == s_end_t2 -1:
					if fp_t2 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t2,"pure",src_t2,image_t2,fp_t2)
						fp_t2 =fp_t2 +1
					elif fp_t2 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t2,"pure",src_t2,image_t2,fp_t2)
						fp_t2 = 0
						image_t2 = s_start_t2
						src_t2 = src_t2 +1
			elif src_t2 ==src-1:
				if image_t2 < s_end_t2 -1:
					if fp_t2 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t2,"pure",src_t2,image_t2,fp_t2)
						fp_t2 =fp_t2 +1
					elif fp_t2 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t2,"pure",src_t2,image_t2,fp_t2)
						fp_t2 = 0
						image_t2 =image_t2 +1
				elif image_t2 == s_end_t2 -1:
					if fp_t2 < fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t2,"pure",src_t2,image_t2,fp_t2)
						fp_t2 =fp_t2 +1
					elif fp_t2 == fpc-1:
						filename,filename2 = filename_maker_tif(qfm_t2,"pure",src_t2,image_t2,fp_t2)
						fp_t2 = 0
						image_t2 = s_start_t2
						src_t2 =0
						qfm_t2 =0
		
	
	
		#filename = "sampleimage/Pre_QF/QF_100/seamcarving10_l/ucid00001.jpg"
		#filename2 = "sampleimage/Pre_QF/QF_100/seamcarving10_l_txt/ucid00001.txt"
		
		th1 = Process(target=read_from_disk, args=(filename,filename2,return_dict,i))
		jobs.append(th1)
		th1.start()
	
	for proc in jobs:
		proc.join()
	
	return_temp = return_dict.values()
	for i in range(batch_size) :
		x_temp,y_temp = return_temp[i]
		x.append(x_temp)
		y.append(y_temp)
	
	# np.testing.assert_array_equal(x,result_x3,err_msg='assert fail')
	# np.testing.assert_array_equal(y,result_y3,err_msg='assert fail')
	
	return x, y
	
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
	
def adj_lr(i):
	global STEP_PER_EPOCH_s
	global NUM_EPOCHS_PER_DECAY
	global INITIAL_LEARNING_RATE
	global LEARNING_RATE_DECAY_FACTOR
	global lr
	# Variables that affect learning rate.
	num_batches_per_epoch = STEP_PER_EPOCH_s
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,i,decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
	return lr

def image_standar(xs):## write thread
	xs_tensor= [[[[0 for i in range(3)] for j in range(subimage_size)]for k in range(subimage_size)]for l in range(batch_size)]
	for i in range(0,batch_size):
		xs_tensor[i] = tf.image.per_image_standardization(tf.convert_to_tensor(xs[i]))
	return xs_tensor
	
def clip_gradients(gradients):
	gradients_t = gradients
	for i in range(0,len(gradients)):
		gradients_t[i] = np.nan_to_num(np.array(gradients[i]),False).tolist()
	return gradients_t
	

# define placeholder for inputs to network
with tf.device('/cpu:0'):
	xs = tf.placeholder(tf.float16, [batch_size,subimage_size, subimage_size,3])
	ys = tf.placeholder(tf.float16, [batch_size,subimage_size//label_scale, subimage_size//label_scale])
	 

## prepare sample ##
with tf.device('/device:GPU:1'):#'/device:GPU:0'
	xs_tensor = image_standar(xs)
	x_image = tf.cast(tf.reshape(xs_tensor, [-1, 512, 512, 3]),tf.float16)
	

## conv1 layer ##
with tf.device('/device:GPU:0'):
	W_conv1 = weight_variable_xavier([5,5, 3,32],"W_conv1") # patch 5x5, in size 1, out size 32
	b_conv1 = bias_variable([32],"b_conv1")
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1,"h_conv1") + b_conv1, name="h_conv1") # output size 512x512x32
	h_pool1 = avg_pool_2x2(h_conv1,"h_pool1") # output size 256x256x32
	# h_pool1_LRN = tf.nn.lrn(h_pool1,name="h_pool1_LRN")# output size 256x256x32
	# W_conv1_all = tf.summary.histogram("W_conv1",W_conv1)
	# b_conv1_all = tf.summary.histogram("b_conv1",b_conv1)
	
## conv2 layer ##
with tf.device('/device:GPU:0'):
	W_conv2 = weight_variable_xavier([5,5, 32, 64],"W_conv2") # patch 5x5, in size 32, out size 64
	b_conv2 = bias_variable([64],"b_conv2")
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,"h_conv2") + b_conv2, name="h_conv2") # output size 256x256x64
	h_pool2 = avg_pool_2x2(h_conv2,"h_pool2") # output size 128x128x64
	# h_pool2_LRN = tf.nn.lrn(h_pool2,name="h_pool2_LRN")# output size 128x128x64
	# W_conv2_all = tf.summary.histogram("W_conv2",W_conv2)
	# b_conv2_all = tf.summary.histogram("b_conv2",b_conv2)
	
## conv3 layer ##
with tf.device('/device:GPU:0'):
	W_conv3 = weight_variable_xavier([5,5, 64, 128],"W_conv3") # patch 5x5, in size 64, out size 128
	b_conv3 = bias_variable([128],"b_conv3")
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3,"h_conv3") + b_conv3, name="h_conv3") # output size 128x128x128
	h_pool3 = avg_pool_2x2(h_conv3,"h_pool3") # output size 64x64x128
	# h_pool3_LRN = tf.nn.lrn(h_pool3,name="h_pool3_LRN")# output size 64x64x128
	# W_conv3_all = tf.summary.histogram("W_conv3",W_conv3)
	# b_conv3_all = tf.summary.histogram("b_conv3",b_conv3)
	
## conv4 layer ##
with tf.device('/device:GPU:0'):
	W_conv4 = weight_variable_xavier([5,5, 128, 256],"W_conv4") # patch 5x5, in size 128, out size 256
	b_conv4 = bias_variable([256],"b_conv4")
	h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4,"h_conv4") + b_conv4, name="h_conv4") # output size 64*64*256
	h_pool4 = avg_pool_2x2(h_conv4,"h_pool4") # output size 32x32x256
	# h_pool4_LRN = tf.nn.lrn(h_pool4,name="h_pool4_LRN")# output size 32x32x256
	# W_conv4_all = tf.summary.histogram("W_conv4",W_conv4)
	# b_conv4_all = tf.summary.histogram("b_conv4",b_conv4)
	
## conv5 layer ##
with tf.device('/device:GPU:0'):
	W_conv5 = weight_variable_xavier([5,5, 256, 512],"W_conv5") # patch 5x5, in size 256, out size 512
	b_conv5 = bias_variable([512],"b_conv5")
	h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5,"h_conv5") + b_conv5, name="h_conv5") # output size 32*32*512
	h_pool5 = avg_pool_2x2(h_conv5,"h_pool5") # output size 16x16x512
	# h_pool5_LRN = tf.nn.lrn(h_pool5,name="h_pool5_LRN")# output size 16x16x512
	# W_conv5_all = tf.summary.histogram("W_conv5",W_conv5)
	# b_conv5_all = tf.summary.histogram("b_conv5",b_conv5)
	
## conv6 layer ##
with tf.device('/device:GPU:0'):
	W_conv6 = weight_variable_xavier([5,5, 512, 1024],"W_conv6") # patch 5x5, in size 256, out size 1024
	b_conv6 = bias_variable([1024],"b_conv6")
	h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6,"h_conv6") + b_conv6, name="h_conv6") # output size 16*16*1024
	h_pool6 = avg_pool_2x2(h_conv6,"h_pool6") # output size 8x8x1024
	# h_pool6_LRN = tf.nn.lrn(h_pool6,name="h_pool6_LRN")# output size 8x8x1024
	# W_conv6_all = tf.summary.histogram("W_conv6",W_conv6)
	# b_conv6_all = tf.summary.histogram("b_conv6",b_conv6)

	
# ## conv7 layer ##
# with tf.device('/device:GPU:0'):
	# W_conv7 = weight_variable_xavier([5,5, 1024, 2048],"W_conv7") # patch 5x5, in size 256, out size 1024
	# b_conv7 = bias_variable([2048],"b_conv7")
	# h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7,"h_conv7") + b_conv7, name="h_conv7") # output size 8*8*2048
	# h_pool7 = avg_pool_2x2(h_conv7,"h_pool7") # output size 4x4x2048
	# # h_pool7_LRN = tf.nn.lrn(h_pool7,name="h_pool7_LRN")# output size 4x4x2048
	# # W_conv7_all = tf.summary.histogram("W_conv7",W_conv7)
	# # b_conv7_all = tf.summary.histogram("b_conv7",b_conv7)
	
# with tf.device('/device:GPU:0'):
	# W_conv8 = weight_variable_xavier([4,4, 2048, 4096],"W_conv8") # patch 5x5, in size 256, out size 1024
	# b_conv8 = bias_variable([4096],"b_conv8")
	# h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8,"h_conv8") + b_conv8, name="h_conv8") # output size 4*4*4096
	# h_pool8 = avg_pool_2x2(h_conv8,"h_pool8") # output size 2*2*4096
	# # h_pool8_LRN = tf.nn.lrn(h_pool8,name="h_pool8_LRN")# output size 1*1*4096
	# # W_conv8_all = tf.summary.histogram("W_conv8",W_conv8)
	# # b_conv8_all = tf.summary.histogram("b_conv8",b_conv8)	
	
	# t_v = tf.trainable_variables()
	
## fc1 layer ##
with tf.device('/device:GPU:0'):
	W_fc1 = weight_variable([8*8*1024, 14000],"W_fc1")#65536,30000
	b_fc1 = bias_variable([14000],"b_fc1")
	# [n_samples, 125, 125, 128] ->> [n_samples, 125*125*512]
	h_pool6_flat = tf.reshape(h_pool6, [-1, 8*8*1024])
	h_fc1 = tf.nn.tanh(tf.matmul(h_pool6_flat, W_fc1) + b_fc1,name="h_fc1")#tf.nn.relu
	h_fc1_drop = tf.cast(tf.nn.dropout(tf.cast(h_fc1,tf.float32), KEEP_PROB_SET),tf.float16, name="h_fc1_drop")
	h_fc1_drop_1 = tf.where(tf.is_nan(h_fc1_drop), tf.constant(min_clip,dtype=tf.float16, shape=h_fc1_drop.shape), h_fc1_drop)
	# W_fc1_all = tf.summary.histogram("W_fc1",W_fc1)
	# b_fc1_all = tf.summary.histogram("b_fc1",b_fc1)

# ##fully connect fc2 layer##
# with tf.device('/device:GPU:0'):
	# W_fc2 = weight_variable([3000,3000],"W_fc2")
	# b_fc2 = bias_variable([3000],"b_fc2")
	# h_fc2 = tf.nn.tanh(tf.matmul(h_fc1_drop_1, W_fc2) + b_fc2, name="h_fc2")#tf.nn.relu
	# h_fc2_1 = tf.where(tf.is_nan(h_fc2), tf.constant(min_clip,dtype=tf.float16, shape=h_fc2.shape), h_fc2)
	# # W_fc2_all = tf.summary.histogram("W_fc2",W_fc2)
	# # b_fc2_all = tf.summary.histogram("b_fc2",b_fc2)
	
# ##fully connect fc3 layer##
# with tf.device('/device:GPU:0'):
	# W_fc3 = weight_variable([3000,3000],"W_fc3")
	# b_fc3 = bias_variable([3000],"b_fc3")
	# h_fc3 = tf.nn.tanh(tf.matmul(h_fc2_1, W_fc3) + b_fc3, name="h_fc3")#tf.nn.relu
	# h_fc3_1 = tf.where(tf.is_nan(h_fc3), tf.constant(min_clip,dtype=tf.float16, shape=h_fc3.shape), h_fc3)
	# # W_fc3_all = tf.summary.histogram("W_fc3",W_fc3)
	# # b_fc3_all = tf.summary.histogram("b_fc3",b_fc3)
	
##down sampleing fc4 layer##
with tf.device('/device:GPU:0'):
	W_fc4 = weight_variable([14000,((subimage_size//label_scale)//2)],"W_fc4")
	b_fc4 = bias_variable([((subimage_size//label_scale)//2)],"b_fc4")
	h_fc4 = tf.nn.tanh(tf.matmul(h_fc1_drop_1, W_fc4) + b_fc4, name="h_fc4")#tf.nn.relu h_fc3_1
	h_fc4_1 = tf.where(tf.is_nan(h_fc4), tf.constant(min_clip,dtype=tf.float16, shape=h_fc4.shape), h_fc4)
	# W_fc4_all = tf.summary.histogram("W_fc4",W_fc4)
	# b_fc4_all = tf.summary.histogram("b_fc4",b_fc4)
	
## fc5 layer method 2##
with tf.device('/device:GPU:0'):
	W_fc5 = weight_variable([((subimage_size//label_scale)//2), 1500],"W_fc5")
	b_fc5 = bias_variable([1500],"b_fc5")
	h_fc5 = tf.nn.tanh(tf.matmul(h_fc4_1, W_fc5) + b_fc5, name="h_fc5")#tf.nn.relu
	h_fc5_1 = tf.where(tf.is_nan(h_fc5), tf.constant(min_clip,dtype=tf.float16, shape=h_fc5.shape), h_fc5)
	# W_fc5_all = tf.summary.histogram("W_fc5",W_fc5)
	# b_fc5_all = tf.summary.histogram("b_fc5",b_fc5)
	
## fc6 layer method 2##
with tf.device('/device:GPU:0'):
	W_fc6 = weight_variable([1500, (subimage_size//label_scale)*(subimage_size//label_scale)],"W_fc6")#262,144
	b_fc6 = bias_variable([(subimage_size//label_scale)*(subimage_size//label_scale)],"b_fc6")
	prediction = tf.nn.tanh(tf.matmul(h_fc5_1, W_fc6)*10 + b_fc6, name="prediction") #tf.nn.sigmoid
	prediction_1 = tf.where(tf.is_nan(prediction), tf.constant(0.0,dtype=tf.float16, shape=prediction.shape), prediction)
	# W_fc6_all = tf.summary.histogram("W_fc6",W_fc6)
	# b_fc6_all = tf.summary.histogram("b_fc6",b_fc6)
	# prediction_all = tf.summary.image("prediction",tf.reshape(prediction_1*255,[batch_size,subimage_size,subimage_size,1]),max_outputs=batch_size)
	#tf.clip_by_value(y_conv,1e-10,1.0)
	
with tf.device('/device:GPU:1'):
# pre progress y
	# ys_down = downsample(ys)
	# y_reshape = tf.reshape(ys_down, [-1, (subimage_size/2)*(subimage_size/2)])
	y_reshape = tf.reshape(ys, [-1, (subimage_size//label_scale)*(subimage_size//label_scale)])

#the error between prediction and real data
	loss = tf.losses.mean_squared_error(y_reshape,prediction_1)
	#loss = tf.losses.mean_pairwise_squared_error(y_reshape,prediction_1)
	loss_sum = tf.summary.scalar("loss", loss)

with tf.device('/device:GPU:0'):	
	# t_v = tf.trainable_variables()
	train_step = tf.train.AdamOptimizer(lr,name='Adam').minimize(loss)##, var_list =t_v
	# optimizer = tf.train.AdamOptimizer(lr,name='Adam')
	# gradients, variables = zip(*optimizer.compute_gradients(loss))##, var_list =t_v
	# gradients_1 = [None if gradient is None else tf.clip_by_norm(gradient, tf.norm(gradient,name="norm"))for gradient in gradients]
	# train_step = optimizer.apply_gradients(zip(gradients_1, variables))
	# gradients_2 = clip_gradients(gradients_1)
	# train_step = optimizer.apply_gradients(zip(gradients_2, variables))
	
##set  dataset reader iterator
xx = [1]
yy = [1]

# dataset1 = tf.data.Dataset.from_tensor_slices((xx,yy))
# dataset1 = dataset1.map(lambda x,y: tuple(tf.py_func( read_train, [x,y], [tf.int64, tf.float32]))).repeat()
# iterator1 = dataset1.make_one_shot_iterator()
# next_element1 = iterator1.get_next()

dataset2 = tf.data.Dataset.from_tensor_slices((xx,yy))
dataset2 = dataset2.map(lambda x,y: tuple(tf.py_func( read_test, [x,y], [tf.int64, tf.float64]))).repeat()
iterator2 = dataset2.make_one_shot_iterator()
next_element2 = iterator2.get_next()

dataset3 = tf.data.Dataset.from_tensor_slices((xx,yy))
dataset3 = dataset3.map(lambda x,y: tuple(tf.py_func( read_test_2, [x,y], [tf.int64, tf.float64]))).repeat()
iterator3 = dataset3.make_one_shot_iterator()
next_element3 = iterator3.get_next()
lrs = tf.placeholder(tf.int32)
adj_lrm =adj_lr(lrs)

# def print_tensor():
# 	global W_conv1
# 	return W_conv1
# print_t = print_tensor()
W_conv1_rng = tf.where(tf.is_nan(W_conv1), tf.constant(min_clip,dtype=tf.float16, shape=W_conv1.shape), W_conv1)
b_conv1_rng = tf.where(tf.is_nan(b_conv1), tf.constant(min_clip,dtype=tf.float16, shape=b_conv1.shape), b_conv1)
W_conv2_rng = tf.where(tf.is_nan(W_conv2), tf.constant(min_clip,dtype=tf.float16, shape=W_conv2.shape), W_conv2)
b_conv2_rng = tf.where(tf.is_nan(b_conv2), tf.constant(min_clip,dtype=tf.float16, shape=b_conv2.shape), b_conv2)
W_conv3_rng = tf.where(tf.is_nan(W_conv3), tf.constant(min_clip,dtype=tf.float16, shape=W_conv3.shape), W_conv3)
b_conv3_rng = tf.where(tf.is_nan(b_conv3), tf.constant(min_clip,dtype=tf.float16, shape=b_conv3.shape), b_conv3)
W_conv4_rng = tf.where(tf.is_nan(W_conv4), tf.constant(min_clip,dtype=tf.float16, shape=W_conv4.shape), W_conv4)
b_conv4_rng = tf.where(tf.is_nan(b_conv4), tf.constant(min_clip,dtype=tf.float16, shape=b_conv4.shape), b_conv4)
W_conv5_rng = tf.where(tf.is_nan(W_conv5), tf.constant(min_clip,dtype=tf.float16, shape=W_conv5.shape), W_conv5)
b_conv5_rng = tf.where(tf.is_nan(b_conv5), tf.constant(min_clip,dtype=tf.float16, shape=b_conv5.shape), b_conv5)
W_conv6_rng = tf.where(tf.is_nan(W_conv6), tf.constant(min_clip,dtype=tf.float16, shape=W_conv6.shape), W_conv6)
b_conv6_rng = tf.where(tf.is_nan(b_conv6), tf.constant(min_clip,dtype=tf.float16, shape=b_conv6.shape), b_conv6)
# W_conv7_rng = tf.where(tf.is_nan(W_conv7), tf.constant(min_clip,dtype=tf.float16, shape=W_conv7.shape), W_conv7)
# b_conv7_rng = tf.where(tf.is_nan(b_conv7), tf.constant(min_clip,dtype=tf.float16, shape=b_conv7.shape), b_conv7)
# W_conv8_rng = tf.where(tf.is_nan(W_conv8), tf.constant(min_clip,dtype=tf.float16, shape=W_conv8.shape), W_conv8)
# b_conv8_rng = tf.where(tf.is_nan(b_conv8), tf.constant(min_clip,dtype=tf.float16, shape=b_conv8.shape), b_conv8)
W_fc1_rng = tf.where(tf.is_nan(W_fc1), tf.constant(min_clip,dtype=tf.float16, shape=W_fc1.shape), W_fc1)
b_fc1_rng = tf.where(tf.is_nan(b_fc1), tf.constant(min_clip,dtype=tf.float16, shape=b_fc1.shape), b_fc1)
# W_fc2_rng = tf.where(tf.is_nan(W_fc2), tf.constant(min_clip,dtype=tf.float16, shape=W_fc2.shape), W_fc2)
# b_fc2_rng = tf.where(tf.is_nan(b_fc2), tf.constant(min_clip,dtype=tf.float16, shape=b_fc2.shape), b_fc2)
# W_fc3_rng = tf.where(tf.is_nan(W_fc3), tf.constant(min_clip,dtype=tf.float16, shape=W_fc3.shape), W_fc3)
# b_fc3_rng = tf.where(tf.is_nan(b_fc3), tf.constant(min_clip,dtype=tf.float16, shape=b_fc3.shape), b_fc3)
W_fc4_rng = tf.where(tf.is_nan(W_fc4), tf.constant(min_clip,dtype=tf.float16, shape=W_fc4.shape), W_fc4)
b_fc4_rng = tf.where(tf.is_nan(b_fc4), tf.constant(min_clip,dtype=tf.float16, shape=b_fc4.shape), b_fc4)
W_fc5_rng = tf.where(tf.is_nan(W_fc5), tf.constant(min_clip,dtype=tf.float16, shape=W_fc5.shape), W_fc5)
b_fc5_rng = tf.where(tf.is_nan(b_fc5), tf.constant(min_clip,dtype=tf.float16, shape=b_fc5.shape), b_fc5)
W_fc6_rng = tf.where(tf.is_nan(W_fc6), tf.constant(min_clip,dtype=tf.float16, shape=W_fc6.shape), W_fc6)
b_fc6_rng = tf.where(tf.is_nan(b_fc6), tf.constant(min_clip,dtype=tf.float16, shape=b_fc6.shape), b_fc6)

# W_conv1_rng = tf.where(tf.is_inf(W_conv1_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_conv1_rng2.shape), W_conv1_rng2)
# b_conv1_rng = tf.where(tf.is_inf(b_conv1_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_conv1_rng2.shape), b_conv1_rng2)
# W_conv2_rng = tf.where(tf.is_inf(W_conv2_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_conv2_rng2.shape), W_conv2_rng2)
# b_conv2_rng = tf.where(tf.is_inf(b_conv2_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_conv2_rng2.shape), b_conv2_rng2)
# W_conv3_rng = tf.where(tf.is_inf(W_conv3_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_conv3_rng2.shape), W_conv3_rng2)
# b_conv3_rng = tf.where(tf.is_inf(b_conv3_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_conv3_rng2.shape), b_conv3_rng2)
# W_conv4_rng = tf.where(tf.is_inf(W_conv4_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_conv4_rng2.shape), W_conv4_rng2)
# b_conv4_rng = tf.where(tf.is_inf(b_conv4_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_conv4_rng2.shape), b_conv4_rng2)
# W_conv5_rng = tf.where(tf.is_inf(W_conv5_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_conv5_rng2.shape), W_conv5_rng2)
# b_conv5_rng = tf.where(tf.is_inf(b_conv5_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_conv5_rng2.shape), b_conv5_rng2)
# W_conv6_rng = tf.where(tf.is_inf(W_conv6_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_conv6_rng2.shape), W_conv6_rng2)
# b_conv6_rng = tf.where(tf.is_inf(b_conv6_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_conv6_rng2.shape), b_conv6_rng2)
# W_fc1_rng = tf.where(tf.is_inf(W_fc1_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_fc1_rng2.shape), W_fc1_rng2)
# b_fc1_rng = tf.where(tf.is_inf(b_fc1_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_fc1_rng2.shape), b_fc1_rng2)
# W_fc2_rng = tf.where(tf.is_inf(W_fc2_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_fc2_rng2.shape), W_fc2_rng2)
# b_fc2_rng = tf.where(tf.is_inf(b_fc2_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_fc2_rng2.shape), b_fc2_rng2)
# W_fc3_rng = tf.where(tf.is_inf(W_fc3_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_fc3_rng2.shape), W_fc3_rng2)
# b_fc3_rng = tf.where(tf.is_inf(b_fc3_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_fc3_rng2.shape), b_fc3_rng2)
# W_fc4_rng = tf.where(tf.is_inf(W_fc4_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_fc4_rng2.shape), W_fc4_rng2)
# b_fc4_rng = tf.where(tf.is_inf(b_fc4_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_fc4_rng2.shape), b_fc4_rng2)
# W_fc5_rng = tf.where(tf.is_inf(W_fc5_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_fc5_rng2.shape), W_fc5_rng2)
# b_fc5_rng = tf.where(tf.is_inf(b_fc5_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_fc5_rng2.shape), b_fc5_rng2)
# W_fc6_rng = tf.where(tf.is_inf(W_fc6_rng2), tf.constant(max_clip,dtype=tf.float16, shape=W_fc6_rng2.shape), W_fc6_rng2)
# b_fc6_rng = tf.where(tf.is_inf(b_fc6_rng2), tf.constant(max_clip,dtype=tf.float16, shape=b_fc6_rng2.shape), b_fc6_rng2)

W_conv1_rns = tf.placeholder(tf.float16, [5,5, 3,32])
b_conv1_rns = tf.placeholder(tf.float16, [32])
W_conv2_rns = tf.placeholder(tf.float16, [5,5, 32, 64])
b_conv2_rns = tf.placeholder(tf.float16, [64])
W_conv3_rns = tf.placeholder(tf.float16, [5,5, 64, 128])
b_conv3_rns = tf.placeholder(tf.float16, [128])
W_conv4_rns = tf.placeholder(tf.float16, [5,5, 128, 256])
b_conv4_rns = tf.placeholder(tf.float16, [256])
W_conv5_rns = tf.placeholder(tf.float16, [5,5, 256, 512])
b_conv5_rns = tf.placeholder(tf.float16, [512])
W_conv6_rns = tf.placeholder(tf.float16, [5,5, 512, 1024])
b_conv6_rns = tf.placeholder(tf.float16, [1024])
# W_conv7_rns = tf.placeholder(tf.float16, [5,5, 1024, 2048])
# b_conv7_rns = tf.placeholder(tf.float16, [2048])
# W_conv8_rns = tf.placeholder(tf.float16, [4,4, 2048, 4096])
# b_conv8_rns = tf.placeholder(tf.float16, [4096])
W_fc1_rns = tf.placeholder(tf.float16, [8*8*1024, 14000])
b_fc1_rns = tf.placeholder(tf.float16, [14000])
# W_fc2_rns = tf.placeholder(tf.float16, [3000,3000])
# b_fc2_rns = tf.placeholder(tf.float16, [3000])
# W_fc3_rns = tf.placeholder(tf.float16, [3000,3000])
# b_fc3_rns = tf.placeholder(tf.float16, [3000])
W_fc4_rns = tf.placeholder(tf.float16, [14000,((subimage_size//label_scale)//2)])
b_fc4_rns = tf.placeholder(tf.float16, [((subimage_size//label_scale)//2)])
W_fc5_rns = tf.placeholder(tf.float16, [((subimage_size//label_scale)//2), 1500])
b_fc5_rns = tf.placeholder(tf.float16, [1500])
W_fc6_rns = tf.placeholder(tf.float16, [1500, (subimage_size//label_scale)*(subimage_size//label_scale)])
b_fc6_rns = tf.placeholder(tf.float16, [(subimage_size//label_scale)*(subimage_size//label_scale)])

W_conv1_rn = tf.assign(W_conv1,W_conv1_rns)
b_conv1_rn = tf.assign(b_conv1,b_conv1_rns)
W_conv2_rn = tf.assign(W_conv2,W_conv2_rns)
b_conv2_rn = tf.assign(b_conv2,b_conv2_rns)
W_conv3_rn = tf.assign(W_conv3,W_conv3_rns)
b_conv3_rn = tf.assign(b_conv3,b_conv3_rns)
W_conv4_rn = tf.assign(W_conv4,W_conv4_rns)
b_conv4_rn = tf.assign(b_conv4,b_conv4_rns)
W_conv5_rn = tf.assign(W_conv5,W_conv5_rns)
b_conv5_rn = tf.assign(b_conv5,b_conv5_rns)
W_conv6_rn = tf.assign(W_conv6,W_conv6_rns)
b_conv6_rn = tf.assign(b_conv6,b_conv6_rns)
# W_conv7_rn = tf.assign(W_conv7,W_conv7_rns)
# b_conv7_rn = tf.assign(b_conv7,b_conv7_rns)
# W_conv8_rn = tf.assign(W_conv8,W_conv8_rns)
# b_conv8_rn = tf.assign(b_conv8,b_conv8_rns)
W_fc1_rn = tf.assign(W_fc1,W_fc1_rns)
b_fc1_rn = tf.assign(b_fc1,b_fc1_rns)
# W_fc2_rn = tf.assign(W_fc2,W_fc2_rns)
# b_fc2_rn = tf.assign(b_fc2,b_fc2_rns)
# W_fc3_rn = tf.assign(W_fc3,W_fc3_rns)
# b_fc3_rn = tf.assign(b_fc3,b_fc3_rns)
W_fc4_rn = tf.assign(W_fc4,W_fc4_rns)
b_fc4_rn = tf.assign(b_fc4,b_fc4_rns)
W_fc5_rn = tf.assign(W_fc5,W_fc5_rns)
b_fc5_rn = tf.assign(b_fc5,b_fc5_rns)
W_fc6_rn = tf.assign(W_fc6,W_fc6_rns)
b_fc6_rn = tf.assign(b_fc6,b_fc6_rns)

save_timeg = tf.Variable(0.0,tf.float32)
# save_res1g = tf.Variable(0.0,tf.float32)
save_res2g = tf.Variable(0.0,tf.float32)
save_times =tf.placeholder(tf.float32)
# save_res1s = tf.placeholder(tf.float32)
save_res2s = tf.placeholder(tf.float32)
save_time = tf.assign(save_timeg,save_times)
# save_res1 = tf.assign(save_res1g,save_res1s)
save_res2 = tf.assign(save_res2g,save_res2s)
tf.summary.scalar("time per step", save_timeg)
# tf.summary.scalar("loss_t1", save_res1g)
tf.summary.scalar("loss_t2", save_res2g)
 
merged_summary = tf.summary.merge_all()
##setting
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.allow_soft_placement=True
# config.log_device_placement=True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
writer = tf.summary.FileWriter("logs/",sess.graph)
print("start initialization !!")
start = time.time()
sess.run(init)
saver = tf.train.Saver()
sess.graph.finalize()
end = time.time()
elapsed = end - start
print("graph initialization done !! Time taken: ", elapsed, "seconds.")
print("start training !!")
start = time.time()



def remove_nan():
	sess.run(W_conv1_rn,feed_dict = {W_conv1_rns: sess.run(W_conv1_rng)})
	sess.run(b_conv1_rn,feed_dict = {b_conv1_rns: sess.run(b_conv1_rng)})
	sess.run(W_conv2_rn,feed_dict = {W_conv2_rns: sess.run(W_conv2_rng)})
	sess.run(b_conv2_rn,feed_dict = {b_conv2_rns: sess.run(b_conv2_rng)})
	sess.run(W_conv3_rn,feed_dict = {W_conv3_rns: sess.run(W_conv3_rng)})
	sess.run(b_conv3_rn,feed_dict = {b_conv3_rns: sess.run(b_conv3_rng)})
	sess.run(W_conv4_rn,feed_dict = {W_conv4_rns: sess.run(W_conv4_rng)})
	sess.run(b_conv4_rn,feed_dict = {b_conv4_rns: sess.run(b_conv4_rng)})
	sess.run(W_conv5_rn,feed_dict = {W_conv5_rns: sess.run(W_conv5_rng)})
	sess.run(b_conv5_rn,feed_dict = {b_conv5_rns: sess.run(b_conv5_rng)})
	sess.run(W_conv6_rn,feed_dict = {W_conv6_rns: sess.run(W_conv6_rng)})
	sess.run(b_conv6_rn,feed_dict = {b_conv6_rns: sess.run(b_conv6_rng)})
	# sess.run(W_conv7_rn,feed_dict = {W_conv7_rns: sess.run(W_conv7_rng)})
	# sess.run(b_conv7_rn,feed_dict = {b_conv7_rns: sess.run(b_conv7_rng)})
	# sess.run(W_conv8_rn,feed_dict = {W_conv8_rns: sess.run(W_conv8_rng)})
	# sess.run(b_conv8_rn,feed_dict = {b_conv8_rns: sess.run(b_conv8_rng)})
	sess.run(W_fc1_rn,feed_dict = {W_fc1_rns: sess.run(W_fc1_rng)})
	sess.run(b_fc1_rn,feed_dict = {b_fc1_rns: sess.run(b_fc1_rng)})
	# sess.run(W_fc2_rn,feed_dict = {W_fc2_rns: sess.run(W_fc2_rng)})
	# sess.run(b_fc2_rn,feed_dict = {b_fc2_rns: sess.run(b_fc2_rng)})
	# sess.run(W_fc3_rn,feed_dict = {W_fc3_rns: sess.run(W_fc3_rng)})
	# sess.run(b_fc3_rn,feed_dict = {b_fc3_rns: sess.run(b_fc3_rng)})
	sess.run(W_fc4_rn,feed_dict = {W_fc4_rns: sess.run(W_fc4_rng)})
	sess.run(b_fc4_rn,feed_dict = {b_fc4_rns: sess.run(b_fc4_rng)})
	sess.run(W_fc5_rn,feed_dict = {W_fc5_rns: sess.run(W_fc5_rng)})
	sess.run(b_fc5_rn,feed_dict = {b_fc5_rns: sess.run(b_fc5_rng)})
	sess.run(W_fc6_rn,feed_dict = {W_fc6_rns: sess.run(W_fc6_rng)})
	sess.run(b_fc6_rn,feed_dict = {b_fc6_rns: sess.run(b_fc6_rng)})

def restore_net():
	global restore_step
	try:
		print("Model restoring.")
		saver.restore(sess, "net_backup/save_net.ckpt")
		print("Model restored.")
	except:
		print("no saved network found")
		restore_step =0
		print("start from step : ",0)


def set_rs():
	global restore_step
	if len(sys.argv) >1 :
		if sys.argv[1] == "-rs":
			restore_step = int(sys.argv[2])
			print("start from step : ",sys.argv[2])
			restore_net()
		else :
			restore_step =0
	else :
		restore_step =0


def train_step_epoch(ii):
	global p0
	global qfm_t0
	global qfc_t0
	global src_t0
	global image_t0
	global fp_t0
	global substart
	global restore_step
	
	ii2 = ii*(STEP_PER_EPOCH_s//batch_size)
	
	manager = multiprocessing.Manager()
	return_dict_main_1 = manager.dict()
	
	for i in range((restore_step%(STEP_PER_EPOCH_s//batch_size_old))+ii2,(STEP_PER_EPOCH_s//batch_size)+ii2):
		# substart = time.time()
		batch_xs,batch_ys,qfm_t0,qfc_t0,src_t0,image_t0,fp_t0 = p0
		th1 = Process(target=read_train, args=(return_dict_main_1,0))
		th1.start()
		sess.run(train_step, feed_dict = {xs: batch_xs,ys : batch_ys})
		remove_nan()
		sess.run(train_step, feed_dict = {xs: batch_xs,ys : batch_ys})
		remove_nan()
		# if i % 60 == 0:#STEP_PER_EPOCH_s
			# res1 =0
			# for j in range(int(train_divide//(batch_size//2)//10)):
			# #for j in range(1):
				# p1= sess.run(next_element2)
				# batch_xt,batch_yt  =p1
				# res1 = res1 + sess.run(loss,feed_dict = {xs: batch_xt, ys: batch_yt})
				# js =j+1
			# sess.run(save_res1,feed_dict = {save_res1s: res1/js})
			# print(i,": t1: ",res1/js)
		
		if i % 100 == 0:#100
			res2 =0
			for k in range(2):#int(validate_divide//(batch_size//2)//10)
			#for k in range(1):
				p2= sess.run(next_element3)
				batch_xt,batch_yt =p2
				res2 = res2 + sess.run(loss,feed_dict = {xs: batch_xt, ys: batch_yt})
				ks =k+1
			sess.run(save_res2,feed_dict = {save_res2s: res2/ks})
			print(i,": t2: ",res2/ks)
		subend = time.time()
		
		## save everything ##
		## tensorboard ##
		if i % 50 == 0 :
			sess.run(save_time,feed_dict = {save_times: subend - substart})
			summary = sess.run(merged_summary, feed_dict = {xs: batch_xs,ys : batch_ys})##
			writer.add_summary(summary, i)
			writer.flush()
		## variable ##
		if i % 500 == 0: # 100
			save_path = saver.save(sess, "net11/save_net.ckpt")
			
		## update learning rate ##
		sess.run(adj_lrm,feed_dict = {lrs:i})
		# print("step : ",i,"finish")
		
		substart = time.time()
		th1.join()
		return_temp_1 = return_dict_main_1.values()
		p0 = return_temp_1[0]

		
#debug
# print(read_y("/home/benson/workspace/sampleimage/Pre_QF/QF_75/seamcarving10_l_txt/ucid00001.txt"))
# print(read_y2("/home/benson/workspace/sampleimage/Pre_QF/QF_75/seamcarving10_l_txt/ucid00001.txt"))
# for i in range(0,1) :
	# p0= sess.run(next_element1)
	# batch_xs,batch_ys =p0
	# print(sess.run( ys, feed_dict = {xs: batch_xs,ys : batch_ys})[0])
	# print(np.shape(batch_xs))
	# print(np.shape(batch_ys))
# x,y =read_train(1,1)
# print(np.shape(x))
# print(np.shape(y))
	
	
##validate file name
# for j in range(STEP_PER_EPOCH_t1):
	# p1= sess.run(next_element2)
	# batch_xt,batch_yt  =p1
	# sess.run(x_image,feed_dict = {xs: batch_xt, ys: batch_yt})
	# print(filename)
# for k in range(STEP_PER_EPOCH_t2):
	# p2= sess.run(next_element3)
	# batch_xt,batch_yt  =p2
	# sess.run(x_image,feed_dict = {xs: batch_xt, ys: batch_yt})
	# print(filename)

#train control
set_rs()
# restore_net()
init_reader()
return_dict_main_1 = []
return_dict_main_1.append(0)
substart = time.time()
p0 = read_train(return_dict_main_1,0)
init_reader()
restore_step_outer = restore_step
for ii in range((restore_step_outer//(STEP_PER_EPOCH_s//batch_size_old)),EPOCHS_PRESET):
	train_step_epoch(ii)
	print("epoch finish")
	if restore_step >0 :
		print("restart counter")
		init_reader()
		p0 = p0[0],p0[1],0,0,0,1,0
		restore_step =0



end = time.time()
elapsed = end - start
print("train done !! Time taken: ", elapsed, "seconds.")
save_path = saver.save(sess, "net11/save_net.ckpt")

#mppyplot.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
#mppyplot.show()
# nohup python3 -u seamcarving_detect8.py > nohup.log 2>&1 &
# nohup tensorboard --logdir=:/media/benson/SSD/workspace/logs/ > tesorboard.log 2>&1 &
