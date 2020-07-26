# PROGRAMMER: Steve M 
# DATE CREATED: 11.07.20                                 
# REVISED DATE: 
# PURPOSE: Train a new image classifier network to recognize different species of flowers
#          and save the model as a checkpoint

# Import necessary libraries
import matplotlib.pyplot as plt
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
from PIL import Image
import seaborn as sns
import argparse
import json

# Imports functions created for this program
from get_input_args_train import get_input_args

def main():

    # Function to retrieve command line arguments entered by the user
    in_arg = get_input_args()
    
    #start the timer for loading and training the model
    start_time = time.time()
    
    ## Load data
    image_datasets, dataloaders = load_data(in_arg.data_dir)
    
    # Get model
    model = get_model(in_arg.arch)
        
    # Load model
    model = load_model(model, in_arg.arch, in_arg.hidden_units, in_arg.learning_rate)

    # need better place to store this code
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    
    # Train and validate the model
    train(model, in_arg.epochs, in_arg.learning_rate, criterion, optimizer, dataloaders['training'], dataloaders['validation'],in_arg.gpu, start_time)
   
    # Confirm time to train model
    print(f"Time to train and validate model: {(time.time() - start_time):.3f} seconds")

    # Save checkpoint
    save_checkpoint(in_arg.save_dir, model, optimizer, in_arg.epochs, in_arg.arch, image_datasets, in_arg.learning_rate)

 #################################
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

###########################  
## Get model
def get_model(arch):
    if arch == 'vgg19': #setting model based on vgg19
        model = models.vgg19(pretrained=True)
    elif arch == 'alexnet': #setting model based on alexnet
        model = models.alexnet(pretrained = True)
    return model

#############################
## Load model
def load_model (model, arch, hidden_units, learning_rate):
    # Hyperparameters for our network
    if arch == 'vgg19': #setting model based on vgg19
        input_size = 25088
    elif arch == 'alexnet': #setting model based on alexnet
        input_size = 9216
    output_size = 102

    # update classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units,  bias=True)), #first layer
        ('Relu1', nn.ReLU()), #apply activation function
        ('Dropout1', nn.Dropout(p = 0.3)),
        ('fc2', nn.Linear(hidden_units, output_size, bias=True)), #output layer
        ('output', nn.LogSoftmax(dim=1)) #apply loss function
    ]))  
      
    #don't update weights of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False 
        
    # replace classifier in pre-trained model with the updated version
    model.classifier = classifier
        
    # Set criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)           
    
    print(criterion)
    return model

###########################
##Train Model
def train(model, epochs, learning_rate, criterion, optimizer, train_loader, validate_loader, cpu_gpu, start_time):

    model.train() # Puts model into training mode
    print_every = 40
    steps = 0
    use_gpu = False
    
    # Check to see if GPU is available
    if cpu_gpu == 'GPU':
        if torch.cuda.is_available():
            use_gpu = True
            model.cuda() # use 'GPU' if user chooses GPU and it is available

        else:
            model.cpu() # use 'CPU' if user chooses GPU but GPU not available
    else:
        model.cpu() # use 'CPU' if user chooses not to use GPU
        
    print('Model: ', model)
    
    # Iterate through each training pass based on number of epochs & GPU availability
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in iter(train_loader):
            steps += 1

            if use_gpu:
                images = Variable(images.float().cuda())
                labels = Variable(labels.long().cuda()) 
            else:
                images = Variable(images)
                labels = Variable(labels) 

            optimizer.zero_grad() # Clear the gradients to ensure they aren't accumulated
     
            # Forward and backward passes
            output = model.forward(images) # Forward propogation 
            loss = criterion(output, labels) # Calculates loss
            loss.backward() # Calculates gradient
            optimizer.step() # Updates weights based on gradient & learning rate

            running_loss += loss.item()

            # Validate the model
            if steps % print_every == 0:
                validation_loss, accuracy = validate(model, criterion, validate_loader) # Call validate function

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                    "Training Loss: {:.3f} ".format(running_loss/print_every),
                    "Validation Loss: {:.3f} ".format(validation_loss),
                    "Validation Accuracy: {:.3f}".format(accuracy))
                
##############################################
## Functionto validate the accuracy of the model
def validate(model, criterion, data_loader):
    model.eval() # Puts model into validation mode
    accuracy = 0
    test_loss = 0
    
    for inputs, labels in iter(data_loader):
        if torch.cuda.is_available():
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True) 
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output).data 
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)

#########
def save_checkpoint(save_dir, model, optimizer, epochs, arch, image_datasets, learning_rate):
    ## Save the checkpoint
    model.cpu()

    checkpoint = {
        'epochs': epochs, #Number of epochs
        'arch': arch, #architecture used
        'state_dict': model.state_dict(), #learned paramaters from training
        'class_to_idx': image_datasets['training'].class_to_idx, #mapping of flower class values to flower indices
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate':learning_rate, # learning rate for model
        'classifier': model.classifier} #Latest checkpoint to continue training later  
                
    torch.save(checkpoint, save_dir + '/alexnet_checkpoint.pth')
    print('Checkpoint saved!')
                
    return None

###########################
    
if __name__ == "__main__":
    main()
    