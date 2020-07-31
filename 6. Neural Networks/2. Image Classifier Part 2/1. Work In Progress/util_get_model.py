# PROGRAMMER: Steve M 
# DATE CREATED: 31.07.20                                 
# REVISED DATE: 
# PURPOSE: Utility functions to load data and process images

# Import necessary libraries
import torchvision.models as models

## Get model
def get_model(arch):
    if arch == 'vgg19': #setting model based on vgg19
        model = models.vgg19(pretrained=True)
    elif arch == 'alexnet': #setting model based on alexnet
        model = models.alexnet(pretrained = True)
    return model

########