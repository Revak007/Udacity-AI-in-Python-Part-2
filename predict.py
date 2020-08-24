#First preprocess image, then run it through model to calculate probs of outputs, then find top classes..
#..and convert to actual labels. This is all in two funcs..
import predict_helper
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InLineBackend.figure_format = 'retina'
import json

#First, get arguments and assign
input_args = predict_helper.get_arguments()
imagepath,checkpoint,top_k,cat_to_name,gpu = predict_helper.interpret_args(input_args)

#Then predict image directly(the function processes image and rebuilds model)
top_p, top_categories = predict_helper.predict_image(imagepath,checkpoint,top_k,gpu)

#Now load in the mapping 
with open(cat_to_name, 'r') as f:
    cat_to_name = json.load(f)
    
#Now convert top_categories to actual labels
actual_labels = [cat_to_name[i] for i in top_categories]

#Convert top_p to list
top_p = top_p[0].tolist()

#Then to get actual label..
print('Most likely label and probability is:')
print(top_p[0], actual_labels[0])

imagepath_list = imagepath.split("/")
category_number = imagepath_list[2]
print('Actual label is: {}'.format(cat_to_name[category_number]))
print('Top-k predicted labels and probabilities are:')
print(top_p,actual_labels)


#BELOW IS ALL GRAPHING, BUT ISN'T POSSIBLE IN WORKSPACE
#Then graph image and top_p classes
#FIRST GRAPH IMAGE
#fig = plt.figure()
#plt.subplot(1,2,1)
#ax1 = plt.add_axis()
#image_tensor = predict_helper.process_image(Image.open(imagepath))
#ax1 = predict_helper.imshow(image_tensor,ax = plt)
#Split imagepath by '/' so that we can get the numerical label of class
#imagepath_list = imagepath.split("/")
#category_number = imagepath_list[2]
#ax1.title(cat_to_name[category_number])
#ax1.show() #Need this because we are plotting image directly on axis

#THEN GRAPH PROBS ON SAME PLOT
#plt.subplot(1,2,2)
#ax2 = plt.add_axis()
#y = np.arange(len(actual_labels)) #Number of tick marks for labels (which will be on y axis)
#SINCE we want horizontal graph, we still put y first and top_p second, but do barh instead of just bar
#plt.barh(y,top_p,align = 'center', alpha = 0.5)
#plt.yticks(y,actual_labels) #Label the ticks with actual class names
#WE don't need ax2.show() because we aren't manually plotting graph on axis... plt.barh does that for us I think






