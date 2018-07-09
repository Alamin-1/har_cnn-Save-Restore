# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:26:10 2018

@author: xngu0004
"""

import numpy as np
import random
import imageio
import time
from itertools import chain
from keras.models import Model
from keras.utils import np_utils
import keras
from keras import regularizers
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pandas as pd
import collections
import re
import math
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import io 
from mpl_toolkits.mplot3d import Axes3D

def getMetaDataForSurgeries(surgery_type):
	surgeries_metadata = {}
	file = open(root_dir+surgery_type+'_kinematic\\'+'meta_file_'+surgery_type+'.txt','r')
	for line in file: 
		line = line.strip() ## remove spaces
	
		if len(line)==0: ## if end of file
			break
	
		b = line.split()
		surgery_name = b[0] 
		expertise_level = b[1]
		b = b[2:]
		scores = [int(e) for e in b]
		surgeries_metadata[surgery_name]=(expertise_level,scores)
	return surgeries_metadata

############################# Global setup ###############################
#time 
start_time = time.time()

# Global parameters 
root_dir = './JIGSAWS/'
path_to_configurations = './JIGSAWS/Experimental_setup/'
path_to_results = './JIGSAWS/temp/'
nb_epochs = 2
surgery_type = 'Suturing'
dimensions_to_use = range(0,76)
number_of_dimensions = len(dimensions_to_use)
input_shape = (None,number_of_dimensions) # input is used to specify the value of the second dimension (number of variables) 
input_shapes = [[(None,3),(None,9),(None,3),(None,3),(None,1)],[(None,3),(None,9),(None,3),(None,3),(None,1)],[(None,3),(None,9),(None,3),(None,3),(None,1)],[(None,3),(None,9),(None,3),(None,3),(None,1)]]

# for each manipulator   x,y,z  ,rot matrx, x'y'z' , a'b'g' , angle  , ... same for the second manipulator ...   

mapSurgeryDataBySurgeryName = collections.OrderedDict() # indexes surgery data (76 dimensions) by surgery name 
mapExpertiseLevelBySurgeryName = collections.OrderedDict() # indexes exerptise level by surgery name  
classes = ['N','I','E']

nb_classes = len(classes) # = 3
confusion_matrix = pd.DataFrame(np.zeros(shape = (nb_classes,nb_classes)), index = classes, columns = classes ) # matrix used to calculate the JIGSAWS evaluation
encoder = LabelEncoder() # used to transform labels into binary one hot vectors 

surgeries_metadata = getMetaDataForSurgeries(surgery_type)