# Get input arguments from user for prediction
# Imports python modules
import argparse

def get_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window. This function. This function returns these arguments as an
    ArgumentParser object.
    """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Image Classifier Prediction')
    parser.add_argument('image_path', default='./flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
    parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
    parser.add_argument('--saved_checkpoint', default='alexnet_checkpoint.pth', help='Saved checkpoint location')
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    
    return parser.parse_args()