# PROGRAMMER: Steve M 
# DATE CREATED: 01.08.20                                 
# REVISED DATE: 
# PURPOSE: Utilities file to hold all utilities functions when training image classifier program

# Import necessary libraries
import torch
import time
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import argparse
import json
from PIL import Image

## Load transformed data set
def load_data(data_dir):
    """Load data"""

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
   
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'testing': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    #Load the datasets with ImageFolder
    image_datasets = {
        'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {    
        'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32, shuffle=True),
        'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=32, shuffle=False)
    }
        
    return image_datasets, dataloaders

###############################
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # debug test - print image_path
    #print('debug image_path test: ', image_path)
    #print(type(image_path))
    
    # Process a PIL image for use in a PyTorch model using same image transformation code during training
    img = Image.open(image_path)
    transform_image = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    image_tensor = transform_image(img)
    print('Steve image')
    print(type(image_tensor))
    
    return image_tensor

##############################   
