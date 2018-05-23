import os
import numpy as np
import matplotlib.image as mpimg
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
image_c=1338 #1338
fpc = 0


folder_pool2 =['_l_txt','_l_noise/seamcarving5_h_txt','_l_noise/seamcarving10_h_txt','_l_noise/seamcarving20_h_txt',
			   '_l_noise/seamcarving30_h_txt','_l_noise/seamcarving40_h_txt','_l_noise/seamcarving50_h_txt',
			   '_h_txt','_h_noise/seamcarving5_l_txt','_h_noise/seamcarving10_l_txt','_h_noise/seamcarving20_l_txt',
			   '_h_noise/seamcarving30_l_txt','_h_noise/seamcarving40_l_txt','_h_noise/seamcarving50_l_txt']
	
fpc = len(folder_pool2)
#read_train variable
qfm_t0 = 0
qfc_t0 = 0
s_start_t0 =1
s_end_t0 = 1338+1
image_t0 =s_start_t0
src_t0 =0
fp_t0 =0
def filename_maker(qfm_t,qfc_t,src_t,image_t,fp_t):
	global qfm
	global quality_factor
	global seam_remove
	global noise
	global smooth
	global sharp_Radius
	global sharp_Amount
	global sharp_Threshold
	global folder_pool2
	
	#filename = "sampleimage/Pre_QF/QF_100/seamcarving10_l/ucid00001.jpg"
	#filename2 = "sampleimage/Pre_QF/QF_100/seamcarving10_l_txt/ucid00001.txt"
	
	if len(str(image_t)) ==1:
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid0000"+str(image_t)+".txt")
	elif len(str(image_t)) ==2:
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid000"+str(image_t)+".txt")
	elif len(str(image_t)) ==3:
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid00"+str(image_t)+".txt")
	elif len(str(image_t)) ==4:
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid0"+str(image_t)+".txt")
	return filename2

def filename_maker_tif(qfm_t,qfc_t,src_t,image_t,fp_t):
	global qfm
	global quality_factor
	global seam_remove
	global noise
	global smooth
	global sharp_Radius
	global sharp_Amount
	global sharp_Threshold
	global folder_pool2
	
	#filename = "sampleimage/Pre_QF/QF_100/seamcarving10_l/ucid00001.jpg"
	#filename2 = "sampleimage/Pre_QF/QF_100/seamcarving10_l_txt/ucid00001.txt"
	
	if len(str(image_t)) ==1:
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid0000"+str(image_t)+".txt")
	elif len(str(image_t)) ==2:
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid000"+str(image_t)+".txt")
	elif len(str(image_t)) ==3:
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid00"+str(image_t)+".txt")
	elif len(str(image_t)) ==4:
		filename2 =("sampleimage/"+str(qfm[qfm_t])+"/QF_"+qfc_t+"/seamcarving"+str(seam_remove[src_t])+str(folder_pool2[fp_t])+"/ucid0"+str(image_t)+".txt")
	return filename2

def read_y(filename):
	fileptr_in = open(filename, 'r')
	data_list = []
	while True:
		filein_line = fileptr_in.readline()
		if not filein_line: break
		data_list_sec = []	
		for i in range(len(filein_line)-1):
			data_list_sec.append(filein_line[i])
		data_list.append(data_list_sec)
	
	fileptr_in.close()
	
	width = len(data_list)
	height = len(data_list[0])
	data_list =np.array(data_list)	
	
	np.savetxt(filename, data_list, fmt='%s', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
	
	
def read_train():
	global qfc
	global src
	global image_c
	global fpc
	global quality_factor
	
	global qfm_t0
	global qfc_t0
	global s_start_t0
	global s_end_t0
	global image_t0
	global src_t0
	global fp_t0
	
	#set file name
	if qfm_t0 < 2:
		if qfc_t0 < qfc-1:
			if src_t0 <src-1:
				if image_t0 < s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 =image_t0 +1
				elif image_t0 == s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 = s_start_t0
						src_t0 = src_t0 +1
			elif src_t0 ==src-1:
				if image_t0 < s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 =image_t0 +1
				elif image_t0 == s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 = s_start_t0
						src_t0 =0
						qfc_t0 = qfc_t0 +1
		elif qfc_t0 == qfc-1:
			if src_t0 <src-1:
				if image_t0 < s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 =image_t0 +1
				elif image_t0 == s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 = s_start_t0
						src_t0 = src_t0 +1
			elif src_t0 ==src-1:
				if image_t0 < s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 =image_t0 +1
				elif image_t0 == s_end_t0 -1:
					if fp_t0 < fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 =fp_t0 +1
					elif fp_t0 == fpc-1:
						filename2 = filename_maker(qfm_t0,str(quality_factor[qfc_t0]),src_t0,image_t0,fp_t0)
						fp_t0 = 0
						image_t0 = s_start_t0
						src_t0 =0
						qfc_t0 =0
						qfm_t0 = qfm_t0 +1
	elif qfm_t0 == 2:
		if src_t0 <src-1:
			if image_t0 < s_end_t0 -1:
				if fp_t0 < fpc-1:
					filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
					fp_t0 =fp_t0 +1
				elif fp_t0 == fpc-1:
					filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
					fp_t0 = 0
					image_t0 =image_t0 +1
			elif image_t0 == s_end_t0 -1:
				if fp_t0 < fpc-1:
					filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
					fp_t0 =fp_t0 +1
				elif fp_t0 == fpc-1:
					filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
					fp_t0 = 0
					image_t0 = s_start_t0
					src_t0 = src_t0 +1
		elif src_t0 ==src-1:
			if image_t0 < s_end_t0 -1:
				if fp_t0 < fpc-1:
					filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
					fp_t0 =fp_t0 +1
				elif fp_t0 == fpc-1:
					filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
					fp_t0 = 0
					image_t0 =image_t0 +1
			elif image_t0 == s_end_t0 -1:
				if fp_t0 < fpc-1:
					filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
					fp_t0 =fp_t0 +1
				elif fp_t0 == fpc-1:
					filename2 = filename_maker_tif(qfm_t0,"pure",src_t0,image_t0,fp_t0)
					fp_t0 = 0
					image_t0 = s_start_t0
					src_t0 =0
					qfm_t0 =0
	
	# filename2 = "ucid00001.txt"
	read_y(filename2)

# read_train()
for i in range(1338*9*5*7*2):
	read_train()