import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

def load_data(path="./flowers"):
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

    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    return trainloader, validationloader, testloader

def build_model(arch="vgg19", dropout=0.2):
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
    print("Model successfully built!")
    return model


def train_model(model, trainloader, epoch=5, device="cuda"):
    """Trains the model we have built and the data provided"""
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
    criterion=nn.NLLLoss()
    model.to(device)
    
    for e in range(epoch):
        train_losses = []
        running_loss = 0
        model.train()
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
        
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            train_losses.append(running_loss/len(trainloader))
        else:
            print(f"Training loss: {running_loss}")
    print("Model has finished training!")
    
def save_checkpoint(hidden_layers=[10000,1500], check_name="checkpoint.pth"):
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
    checkpoint = torch.load(path)
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
    model.eval()
    model.to(device)
    img = process_image(image_path)
    img = torch.from_numpy(img)
    img.unsqueeze_(0)
    
    with torch.no_grad():
        probs = model.forward(img.cuda())
        probs = F.softmax(probs, dim=1)
        probs = probs.cpu()
        
    return probs.topk(topk)