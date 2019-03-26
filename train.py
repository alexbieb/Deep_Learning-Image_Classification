# Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import json
import copy
import seaborn as sns
import numpy as np
from PIL import Image
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.autograd import Variable

# Import Argparse for the inputs in the command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', action="store",default='/home/workspace/aipnd-project/flowers')
parser.add_argument('--save_dir',action='store',default='/home/workspace/aipnd-project/checkpoint.pth')
parser.add_argument('--mapping_dir',action='store',default='/home/workspace/aipnd-project/cat_to_name.json')
parser.add_argument('--arch',action='store',default='vgg16')
parser.add_argument('--learning_rate',action='store',default=0.003)
parser.add_argument('--hidden_sizes',action='store',default=[4096])
parser.add_argument('--epochs',action='store',default=6)
parser.add_argument('--gpu',action='store',default=True)
parser.add_argument('--batch_size',action='store',default=64)
args = parser.parse_args()

# Define directories
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    
# Additional article Medium
train_dataset_size = len(train_dataset)
valid_dataset_size = len(valid_dataset)
test_dataset_size = len(test_dataset)
class_names = train_dataset.classes
# JSON
with open(args.mapping_dir, 'r') as f:
        cat_to_name = json.load(f)
        
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# Loading a pre-trained network : 
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else :
    print('Model not trained')
    #Define a new, untrained feed-forward network as a classifier
# Freeze the parameters - updating only the weights of the feed-forward network 
for param in model.parameters():
        param.requires_grad = False

# Hyperparameters for our network
input_size = 25088
hidden_sizes = args.hidden_sizes
output_size = 102

# feed-forward network
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('hidden_layer', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('dropout1',nn.Dropout(p=0.1)),
                          ('exit_layer', nn.Linear(hidden_sizes[0], output_size)),
                          ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device)
    
epochs = args.epochs
print_every = 100
steps = 0
running_loss = 0

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1
        
        images,labels = images.to(device),labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model(images)
        loss = criterion(logps,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0
            
            for images, labels in validloader:
                
                images,labels = images.to(device),labels.to(device)
                
                logps = model(images)
                loss = criterion(logps,labels)
                valid_loss += loss.item()
                
                #calculate our accuracy
                ps = torch.exp(logps)
                top_ps,top_class = ps.topk(1,dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
            print(f"Epoch {epoch+1}/{epochs}.."
                f"Train loss : {running_loss/print_every:.3f}.."
                f"Validation loss: {valid_loss/len(validloader):.3f}.."
                f"Validation accuracy: {accuracy/len(validloader):.3f}")
            
            running_loss = 0
            model.train()
              
            # TODO: Do validation on the test set
model.eval()
test_loss = 0
test_accuracy = 0

for images, labels in testloader:
    images,labels = images.to(device),labels.to(device)
    logps = model(images)
    loss = criterion(logps,labels)
    test_loss += loss.item()
                
    #calculate our accuracy
    ps = torch.exp(logps)
    top_ps,top_class = ps.topk(1,dim=1)
    equality = top_class == labels.view(*top_class.shape)
    test_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
print(f"Test accuracy: {test_accuracy/len(testloader):.3f}")

model.class_to_idx = train_dataset.class_to_idx
checkpoint = {'arch' : args.arch,'input_size' : 25088,'hidden_sizes' : args.hidden_sizes,'output_size': 102,'class_to_idx':model.class_to_idx,'state_dict':model.state_dict()}

torch.save(checkpoint,args.save_dir)