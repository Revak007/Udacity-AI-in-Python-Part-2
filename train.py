import train_helper
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import json

#First, get arguments
in_args = train_helper.parse_arguments()

#Then, read in the arguments
data, checkpoint, arch,lr,hidden1,epochs,gpu = train_helper.read_in_arguments(in_args)

#Then update directory names and create transforms
train_dir = data+'/train'
valid_dir = data+'/valid'
test_dir = data+'/test'

train_transforms, test_transforms = train_helper.create_transforms()

#NOTE: Reducing batch size from 64 to 32 reduces memory usage for each batch (so no cuda memory loss)
# Load the datasets with ImageFolder
train_images = datasets.ImageFolder(train_dir,transform = train_transforms)
valid_images = datasets.ImageFolder(valid_dir, transform = test_transforms)
test_images = datasets.ImageFolder(valid_dir, transform = test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_images,batch_size = 32, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_images, batch_size = 32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_images, batch_size = 32, shuffle = True)



#Now, load in model
model = train_helper.load_model(arch)
#model = train_helper.freeze_and_changeclassifier(model,hidden1)

#---WORKS SO FAR!!! NOTE: Need to define input directory as just flowers (not /flowers)
#Train model- within this function, features are frozen and classifier is redefined
#REMEMBER to give --gpu argument, otherwise it won't run on gpu (I think even if gpu is enabled)
trained_model = train_helper.train_model(model,hidden1, epochs,lr,gpu,trainloader,validloader,arch)

#Save model as checkpoint
train_helper.save_model(trained_model, checkpoint,arch,train_images)

