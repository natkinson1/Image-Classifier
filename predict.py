import argparse
import utils
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

def get_parsed_arguments():
    
    parser = argparse.ArgumentParser(description="Predicting an image")
    #Required
    parser.add_argument("image_path", help="Image path that is being predicted")
    #Optional
    parser.add_argument("-c","--checkpoint", dest="check_name",help="Load a certain model checkpoint from .pth file.", default="checkpoint.pth")
    parser.add_argument("-k","--topk", dest="topk", help="Numbers of classes predicted", default=5)
    parser.add_argument("-g", "--gpu", dest="gpu", help="use gpu to predict image")
                    
    return parser.parse_args()

def main():
    
    
    #Names of the flowers
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    #Load arguments
    args = get_parsed_arguments()
    
    #Load the model used for prediction
    model = utils.load_model(args.check_name)
    
    #Predict an image
    
    probs, names = utils.predict(image_path = args.image_path, model=model, topk=args.topk)
    
    probs = probs.numpy()[0]
    
    utils.display(names, probs)

if __name__ == "__main__":
    main()
