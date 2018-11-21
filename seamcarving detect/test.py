import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as mppyplot

subimage_size =512
filename = "/home/benson/workspace/sampleimage/Pos_QF/QF_25/seamcarving10_h_noise/seamcarving5_l_txt/ucid00001.txt"
# filename = "/home/benson/workspace/ucid00001.txt"
data_list = []
output_list = [[0 for i in range(subimage_size)] for j in range(subimage_size)]
data_list = np.loadtxt(filename, dtype='int', delimiter=" ", ndmin=1)

width = len(data_list)
height = len(data_list[0])

output_list = np.array(output_list)
data_list =np.array(data_list)
output_list[0:width, 0:height] = data_list
output_list[output_list >1] = 0	

mppyplot.imshow(output_list)
mppyplot.show()
# numpy.fromfile
# numpy.loadtxt
