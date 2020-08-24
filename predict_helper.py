from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
    description='This is an AI prediction program',
    )
    
    parser.add_argument("imagepath", help="path to image",)
    parser.add_argument("checkpoint",help = "path to checkpoint",)
    parser.add_argument("--top_k",default = 5, help = "Number of top classes you want to list",)
    parser.add_argument("--category_names",default = 'cat_to_name.json', help = "Numerical category to actual name json file path",)
    parser.add_argument("--gpu", default = False, help = "Use gpu or not for inference/to run the image throuh model",)
    
    return parser.parse_args()

def interpret_args(in_args):
    imagepath = in_args.imagepath
    checkpoint = in_args.checkpoint
    top_k = in_args.top_k
    cat_to_name = in_args.category_names
    gpu = in_args.gpu
    return imagepath,checkpoint,top_k,cat_to_name,gpu


def process_image(image):
    #First resize it- keep aspect ratio but make shorter side 256..
    width,height = image.size
    if width == height:
        new_size = int(256)
        image = image.resize(new_size)
    elif width>height:
        new_size = (int(256*(width/height)), int(256))
        image = image.resize(new_size)
    else:
        new_size = (int(256), int(256*(height/width)))
    
    #Then crop out center (so we want this center piece)
    left = (width-224)/2
    right = (width+224)/2
    upper = (height-224)/2
    lower = (height+224)/2
    image = image.crop((left,upper,right,lower))
    
    #Then convert to values between 0 and 1 (convert into numpy array first)
    np_image = np.array(image)
    np_image = np_image/np.max(np_image)
    
    #Then normalize
    np_image[0] =(np_image[0]-0.485)/0.229
    np_image[1] = (np_image[1]-0.456)/0.224
    np_image[2] = (np_image[2]-0.406)/0.225
    
    #Change dimensions
    np_image = np_image.transpose(2,0,1)
    
    #Convert to FloatTensor and return
    tensor_image = torch.FloatTensor(np_image)
    return tensor_image
    
def rebuild_model(checkpoint_path):
    #First load in checkpoint
    checkpt = torch.load(checkpoint_path, map_location=lambda storage, loc:storage)
    #Then load in model and rebuild it
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    model_options = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}
    model = model_options[checkpt['arch']]
    model.classifier = checkpt['classifier']
    model.load_state_dict = checkpt['state_dict']
    model.class_to_idx = checkpt['class_to_idx']
    return model

def predict_image(image_path,checkpoint_path,topk,gpu):
    if gpu == True:
        device = 'cuda'
    else:
        device = 'cpu'
    with torch.no_grad():
        image_tensor = process_image(Image.open(image_path))
        image_tensor = image_tensor.unsqueeze_(0)
        #Convert to float tensor..
        image_tensor = image_tensor.type(torch.FloatTensor)
        image_tensor.to(device)
        #Rebuild model 
        model = rebuild_model(checkpoint_path)
        model.to(device)
        probs = torch.exp(model(image_tensor))
        top_p, top_class = probs.topk(topk,dim=1)
        top_class = top_class.numpy()
        
        #Now convert the top_class index values into numerical values corresponding with classes
        idx_to_class = {k:v for v,k in model.class_to_idx.items()}
            
        class_num_labels = [idx_to_class[i] for i in top_class[0]] #For some reason, top_class is a list within a list
        
        #Also convert top_p to numpy array
        top_p = top_p.numpy()
        return top_p, class_num_labels
    
    
   
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax  
    
   
        
