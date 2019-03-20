import argparse
import utils
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def get_parsed_arguments():
    
    parser = argparse.ArgumentParser(description="Training our model")
    #Required
    parser.add_argument("data_dir", help="Directory of images", default="./flowers/")
    #Optional
    parser.add_argument("-e", "--epoch", dest="epoch",help="Number of epochs to train the data.", default=10, type=int)
    parser.add_argument("-a","--arch", type=str, dest="arch", help="Architecture of network", default="vgg19")
    parser.add_argument("-g","--gpu", type=str, dest="device", help="Choose whether you want to train using GPU.", default="cuda")
    parser.add_argument("-s","--save_dir", dest="save_dir", help="save the directory")
    parser.add_argument("-d","--dropout", dest="dropout", help="dropout percentage for training neural network 0-1")
    parser.add_argument("-hl","--hidden_layers", dest="hidden_layers", help="choose how many hidden layers you want, must match with architecture",
                        default=[10000,1500])
    parser.add_argument("-c","--checkpoint_name", dest="check_name", help="Save model checkpoint", default="checkpoint.pth")
    parser.add_argument("-concheck", "--continue_checkpoint", dest="con_check", help="continue training model from last checkpoint")
    
    return parser.parse_args()

def main():
    
    """Runs the script."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on the: " + str(device))
    
    args = get_parsed_arguments()
    
    #Loading the data
    imagedatasets, dataloader = utils.load_data(path=args.data_dir)
    
    #Build the model
    model = utils.build_model(arch=args.arch, dropout=args.dropout, con_check=args.con_check)
    
    #Train model
    print("Training Model...")
    utils.train_model(model, dataloader["training"], dataloader["validation"], epoch=args.epoch, device=args.device)
    
    #Save model
    print("Saving Model")
    utils.save_checkpoint(model = model, train_data = imagedatasets["training"], check_name=args.check_name)
    
    print("Process Complete, you can now start predicting!")
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
