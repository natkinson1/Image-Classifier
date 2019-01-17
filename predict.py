import argparse
import utils
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="predicting an image")
parser.add_argument("image_path", help="Image that is being predicted")
parser.add_argument("--topk", dest="topk", default=5, help="Numbers of classes predicted")
parser.add_argument("checkpoint", help="Load a checkpoint during training.")
parser.add_argument("--gpu", dest="gpu", help="use gpu to predict image")
                    
parse = parser.parse_args()
image_path = parse.image_path
check_name = parse.checkpoint
gpu = parse.gpu
topk = parse.topk
                    

model = utils.load_model(check_name)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
    
probs, classes = utils.predict(image_path, model, topk=topk)
names = [cat_to_name[str(i+1)] for i in classes.numpy()[0]]

for i in names:
    print("Flower type: {}, with probability: {}.".format(names[i], probs.numpy[0][i]))