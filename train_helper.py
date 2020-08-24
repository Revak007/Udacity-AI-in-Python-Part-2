import argparse
from pathlib import Path
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from workspace_utils import active_session


def parse_arguments():
    '''This function defines argparse arguments'''
    #parser = argparse.ArgumentParser(description = ‘This program takes data, several other hyperparameters, then trains a neural network model on the training data’,)
    parser = argparse.ArgumentParser(
    description='This is an AI training program',
    )
    
    parser.add_argument("filepath", help="path to data directory",)
    #ABOVE- NOT SURE HOW TO READ IN FILE WITHOUT HAVING --ARGUMENT… 
    
    parser.add_argument("--save_dir", type = str, default = 'checkpoint.pth', help = 'Name of file to save checkpoint as')
    parser.add_argument("--arch", type = str, default = 'vgg', help = 'Type of model architecture to use for transfer learning')
    parser.add_argument("--learning_rate", type = int, default  = 0.0003, help = 'Learning rate when training model')
    
    parser.add_argument("--hidden_units", type = int, default = 500, help = 'Size of first hidden layer')
    #ABOVE: I’m just treating it as size of first hidden layer, like stated in the mentor help section. ALSO.. DEFAULT IS 4096 AS PER 60% ACCURATE MODEL
    parser.add_argument("--epochs", type = int, default = 2, help = 'Number of epochs')
    #ABOVE: DEFAULT IS 13 AS PER 60% ACCURATE MODEL
    parser.add_argument("--gpu",type = bool, default = False, help = 'Whether or not we are using gpu')
   
    #NOW, return parsed arguments
    return parser.parse_args()
    
def read_in_arguments(in_args):
    '''Takes in arguments and assigns them to variables to be returned to main program'''
    data = in_args.filepath
    checkpoint = in_args.save_dir
    arch = in_args.arch
    lr = in_args.learning_rate
    hidden1 = in_args.hidden_units
    epochs = in_args.epochs
    gpu_or_not = in_args.gpu
    return data,checkpoint,arch,lr,hidden1,epochs,gpu_or_not

def create_transforms():
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return train_transform, test_transforms

def load_model(arch):
    #We are only giving the user three possibilities
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    model_options = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}
    model = model_options[arch]
    return model

def changeclassifier(model,hidden1,arch):
        
    #THE number of input features will be different based on what model it is!
    if arch == 'vgg':
        input_size = 25088
    elif arch == 'resnet':
        input_size = 512
    elif arch == 'alexnet':
        input_size = 9216
        
    classifier = nn.Sequential(nn.Linear(input_size,hidden1), nn.ReLU(), nn.Dropout(p=0.2), 
                               nn.Linear(hidden1,102), nn.LogSoftmax(dim=1))
    model.classifier = classifier
    return model,input_size

def train_model(model,hidden1,epoch,lr,gpu,trainloader,validloader,arch):
    with active_session():
        if gpu == True:
            device = "cuda"
        else:
            device = "cpu"
        #First freeze feature parameters
        for param in model.parameters():
            param.require_grad = False    
         
        #Then redefine model classifier
        model,input_size = changeclassifier(model,hidden1,arch)
      
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
            
        running_loss = 0
        steps = 0
        print_every = 30
        model.to(device)
        for e in range(epoch):
            for images,labels in trainloader:
                #Move the images and labels to gpu, and increment steps
                images,labels = images.to(device),labels.to(device)
                steps+=1

                #Find output and use backpropagation to update weights
                #NOTE TO SELF: The backprop technique keeps track of all operations on tensors and then..
                #...loss.backward() defines what gradient we want, and optimizer.step() calculates grad of loss with respect to each 
                #..parameter of classifier of model. All this is done via autograd package in torch (so just need import torch)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()

                running_loss+=loss.item()

            #Every 30 batches of training set, run the whole validation set and ensure that losses are decreasing
                if steps%print_every == 0:
                    accuracy = 0
                    test_loss = 0
                    model.eval()
                    for images,labels in validloader:
                        images,labels = images.to(device),labels.to(device)
                        #First find output and loss
                        out = model(images)
                        loss = criterion(out,labels)
                        test_loss+=loss.item()

                        #Then find actual probabilities of each output class to then calculate accuracy
                        ps = torch.exp(out)
                        top_p, top_class = ps.topk(1,dim=1)
                        equals = top_class==labels.view(*top_class.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor))

                    print(f"Epoch {e+1}/{epoch}.."
                          f"Training Loss: {running_loss/print_every:.3f}.."
                          f"Test Loss: {test_loss/len(validloader):.3f}.."
                          f"Test Accuracy: {accuracy/len(validloader):.3f}")

                    running_loss = 0
                    model.train()
    return model   


def save_model(model,checkpoint,arch,train_images):
    '''Saves model info to file whose path is given by checkpoint'''
    model.class_to_idx = train_images.class_to_idx

    #I'm saving the whole classifier plus state (so weights) plus optimizer and epochs so far if we need to train it again later
    checkpt = {'arch':arch,
               'classifier':model.classifier,
              'dropout':0.2,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}
    torch.save(checkpt, checkpoint)
    
    
