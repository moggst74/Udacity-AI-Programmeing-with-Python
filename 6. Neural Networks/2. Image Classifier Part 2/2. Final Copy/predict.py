# PROGRAMMER: Steve M 
# DATE CREATED: 19.07.20                                 
# REVISED DATE: 
# PURPOSE: Load model checkpoint and rebuild model without having to retrain

# Import necessary libraries
import torch
from torchvision import transforms, models
#from PIL import Image
import argparse
import json

# Imports functions created for this program
from get_input_args_predict import get_input_args
from utilities import process_image

def main():

    # Function to retrieve command line arguments entered by the user
    in_arg = get_input_args()    
    
    #Load pre-trained model from checkpoint
    model = load_checkpoint(in_arg.saved_checkpoint, in_arg.gpu)
    #print(in_arg.saved_checkpoint)
    #print(model)
    
    # Process Image
    processed_image = process_image(in_arg.image_path) 
    
    # Classify Prediction
    top_probs, top_labels, top_flowers = predict(in_arg.image_path, model, in_arg.category_names, in_arg.top_k)
    
###############################    
def load_checkpoint(filepath, cpu_gpu):   
    # Check if GPU or CPU used
    if cpu_gpu == 'gpu':
        checkpoint = torch.load(filepath) #use defalut location as model was trained on GPU
    else:
        checkpoint = torch.load(filepath, map_location='cpu') #remap tensor storage location to CPU

    architecture = checkpoint['arch']
    print('Architecture used: ', architecture)
    
    # validate the model used for training
    if architecture =='vgg19':
        model = models.vgg19(pretrained=True)
    elif architecture =='alexnet':
        model = models.alexnet(pretrained = True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Wrong Model")

    model.class_to_idx = checkpoint['class_to_idx']  #mapping of flower class values to flower indices
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])  #learned paramaters from training
 
    return model

###############################

def predict(image_path, model, category_names, topk):
    #Predict the class (or classes) of an image using a trained deep learning model
    
    # Code to predict the class from an image file   
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image))
    top_probs, top_labs = probs.topk(topk)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_labs[0].numpy()

    top_labels = []
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    top_probs = top_probs[0].detach().numpy() #converts from tensor to nparray
    #print(top_flowers)
    #print(top_probs)
    #print(top_labels)
    #print(topk)
    #print('top flowers length')
    #print(len(top_flowers))
    #print(len(top_probs))
    
    #printing out the top K classes along with associated probabilities
    i=0 # this prints out top k classes and probs as according to user 
    while i < topk:
        print("{} with a probability of {:.3f}%".format(top_flowers[i], top_probs[i]*100))
        #print("{} with a probability of {}".format(top_flowers[i]))
        #print('top_probs: ', top_probs[i])
        i += 1 # cycle through
    
    return top_probs, top_labels, top_flowers
###############################
    
if __name__ == "__main__":
    main()
        