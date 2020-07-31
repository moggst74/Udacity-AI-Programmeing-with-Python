# Get input arguments from user for training
# Imports python modules
import argparse

def get_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window. This function. This function returns these arguments as an
    ArgumentParser object.
    """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('data_dir', type=str, help='Path to training data files')
    parser.add_argument('--arch', type=str, default='vgg19', help='CNN model architecture - vgg19 or alexnet')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', type = str,  default = 'GPU', choices=['GPU','CPU'], help = 'Type GPU or CPU with uppercase')

    return parser.parse_args()
