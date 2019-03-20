import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

class Flower_Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(25088,10000)
        self.fc2 = nn.Linear(10000,1500)
        self.fc3 = nn.Linear(1500,102)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x

def load_data(path):
    """Loads the data for the trainloader, validationloader and testloader respectively. """
    
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485,0.456,0.406],
                                                             std=[0.229,0.224,0.225])])

    image_datasets = {
    'training' : datasets.ImageFolder(train_dir, transform=data_transforms) ,
    'validation': datasets.ImageFolder(valid_dir, transform=data_transforms) ,
    'test' : datasets.ImageFolder(test_dir, transform=data_transforms)
    }

    dataloaders = {
    'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True) ,
    'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32) ,
    'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }
    
    return image_datasets, dataloaders

def build_model(arch="vgg19", dropout=0.2, con_check=None):
    """Builds the model I am going to train"""
    if arch == "vgg19": 
        model = models.vgg19(pretrained=True)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "alexnet":
        model = models.alexnet(pretrained=True)
    else:
        print("Try another model like vgg19, densenet121 or alexnet")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = Flower_Network()
    
    if con_check is not None:
        model = load_model(con_check)

    return model


def train_model(model, trainloader, validationloader, epoch=5, device="cuda"):
    """Trains the model we have built and the data provided"""
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
    criterion=nn.NLLLoss()
    model.to(device)
    
    train_losses = []
    val_losses = []

    for e in range(epoch):
    
        running_loss = 0
        model.train()
    
        for images, labels in trainloader:
          
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
        else:
            val_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in validationloader:
                
                    images, labels = images.to(device), labels.to(device)
                
                    log_ps = model(images)
                    val_loss += criterion(log_ps, labels)
                
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
        
            model.train()
                
        train_losses.append(running_loss/len(trainloader))
        val_losses.append(val_loss/len(validationloader))

        print("Epoch: {}/{}.. ".format(e+1, epoch),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Validation Loss: {:.3f}.. ".format(val_loss/len(validationloader)),
              "Accuracy: {:.3f}".format(accuracy/len(validationloader)))
    
def save_checkpoint(model, train_data, hidden_layers=[10000,1500], check_name="checkpoint.pth"):
    """Saves the model to a checkpoint which can be loaded in the future."""
    
    model.class_to_idx = train_data.class_to_idx
    model_info = {"input_size": 25088,
             "output_size": 102,
             "hidden_layers": hidden_layers,
             "state_dict": model.state_dict(),
             "class_to_idx": model.class_to_idx}

    torch.save(model_info, check_name)
    print("Model successfully saved as: {}".format(check_name))
       
def load_model(path):
    """Loads the model you previously built from a checkpoint. Returns the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.to(device)
    checkpoint = torch.load(path, map_location='cpu')
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485,0.456,0.406],
                                                         std=[0.229,0.224,0.225])])
    trans_img = transform(pil_image)
    return np.array(trans_img)

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
   
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    img = process_image(image_path)
    img = torch.from_numpy(img)
    img.unsqueeze_(0)
    
    with torch.no_grad():
        probs = torch.exp(model.forward(img.cuda()))
        probs = probs.cpu()
    
    probs, classes = probs.topk(topk)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    names = [cat_to_name[str(i+1)] for i in classes.numpy()[0]]
        
    return probs, names

def display(names, probs):
    
    for i in range(len(names)):
        print("Flower:", names[i], "with probability:", probs[i])
