import argparse
import utils
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#Commandline Arguments
parser = argparse.ArgumentParser(description="Training our model")
parser.add_argument("--epoch", dest="epoch", action="store",help="Number of epochs to train the data.", default=1)
parser.add_argument("data_dir", help="Directory of images", default="./flowers/")
parser.add_argument("--arch", type=str, dest="arch", action="store", help="Type of deep network", default="vgg19")
parser.add_argument("--gpu", type=str, dest="device", action="store", help="Choose whether you want to train using GPU.", default="cuda")
parser.add_argument("--save_dir", dest="save_dir", action="store", help="save the directory")
parser.add_argument("--dropout", dest="dropout", action="store", help="dropout percentage for training neural network 0-1")
parser.add_argument("--hidden_layers", dest="hidden_layers", action="store", help="choose how many hidden layers you want, must match with architechture",
                    default=[10000,1500])
parser.add_argument("--checkpoint_name", dest="check_name", action="store", help="Choose the name of the checkpoint", default="checkpoint.pth")

parse = parser.parse_args()
data_dir = parse.data_dir
save_path = parse.save_dir
epoch = parse.epoch
device = parse.device
check_name = parse.check_name
arch = parse.arch
dropout = parse.dropout
hidden_layers = parse.hidden_layers

trainloader, validiationloader, testloader = utils.load_data(data_dir)
model = utils.build_model(arch=arch, dropout=dropout)

#utils.train_model(model=model, trainloader=trainloader, epoch=epoch, device=device)

utils.save_checkpoint(hidden_layers=hidden_layers, check_name=check_name)

print("Model successfully saved")
