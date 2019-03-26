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
parser.add_argument('--data_directory', action="store",default='/home/workspace/aipnd-project')
parser.add_argument('--save_dir',action='store',default='/home/workspace/aipnd-project/checkpoint.pth')
parser.add_argument('--image_path',action='store',default='/home/workspace/aipnd-project/flowers/test/12/image_03994.jpg')
parser.add_argument('--mapping_dir',action='store',default='/home/workspace/aipnd-project/cat_to_name.json')
parser.add_argument('--arch',action='store',default='vgg16')
parser.add_argument('--top_categ',action='store',default=5)
args = parser.parse_args()
print(args)

checkpoint = torch.load(args.save_dir)

if checkpoint['arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)
elif checkpoint['arch'] == 'vgg13':
    model = models.vgg13(pretrained=True)
else :
    print('Model not trained')

for param in model.parameters():
    param.requires_grad = False
    input_size = checkpoint['input_size']
    hidden_sizes = checkpoint['hidden_sizes']
    output_size = checkpoint['output_size']
    model.class_to_idx = checkpoint['class_to_idx']
    
with open(args.mapping_dir, 'r') as f:
        cat_to_name = json.load(f)
        
classifier = nn.Sequential(OrderedDict([
                          ('hidden_layer', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('dropout1',nn.Dropout(p=0.1)),
                          ('exit_layer', nn.Linear(hidden_sizes[0], output_size)),
                          ('output', nn.LogSoftmax(dim=1))]))
    
model.classifier = classifier
    
model.load_state_dict(checkpoint['state_dict'])

image = []
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    # Resizing
    width, height = image.size
    if width < height :
        image = image.resize(size=(256,height))
    else :
        image = image.resize(size=(width,256))
    
    # Cropping by the center
    width, height = image.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
    
    # Numpy array
    image = np.array(image)
    
    # Normalization
    image = image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    
    # Arrange the third dimension in 1st position and keep the 2 others in the same order
    image = np.transpose(image,(2,0,1))
    
    return image

process_image(args.image_path)

proba = []
classes_mapped = []
classes_names = []
topcateg=int(args.top_categ)

def predict(image_path,model=model,topcateg=topcateg):
    model.eval()
    global proba
    global classes_names
    proba = []
    classes_names = []
    image = process_image(image_path)
    image = torch.from_numpy(image).float().to('cpu')
    image = image.unsqueeze(0)
    probs = torch.exp(model.forward(image))
    probs,classes = probs.topk(topcateg)
    proba = probs.data.cpu().numpy().tolist()[0]
    proba = [round(prob, 3) for prob in proba]
    classes = classes.data.cpu().numpy().tolist()[0]
    
    # mapping with the correct labels
    inv_map = {v: k for k, v in model.class_to_idx.items()}
            
    for i in range(0,topcateg):
        top_i = classes[i]
        top_i = inv_map[top_i]
        top_i = cat_to_name[top_i]
        classes_names.append(top_i)
        image_path = args.image_path
        num = image_path.split('/')[6]
        title = cat_to_name[num]
    print(title)
    print(proba)
    print(classes_names)
    
predict(args.image_path)
