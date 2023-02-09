import torchvision.models as models
import argparse
from torch import nn,optim
from collections import OrderedDict
import torch
import torchvision
    
def network_setup_systems(args):
    if torch.cuda.is_available() and args.gpu == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    arch = args.model_arch 
    if(arch =="vgg11"):
        model = models.vgg11(pretrained=True)
    elif(arch == "vgg13"):
        model = models.vgg13(pretrained=True)
    elif(arch == "vgg16"):
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    for param in model.parameters():
        param = False
    classifier = nn.Sequential(OrderedDict([
        ('door',nn.Linear(25088,args.hidden_units,bias=True)),
        ('reLu1',nn.ReLU()),
        ('dropout1',nn.Dropout(0.2)),
        ('out',nn.Linear(args.hidden_units,102)),
        ('f_out',nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = args.learning_rate)
    return model,criterion,optimizer
    

def save_model_func(args,model,optimizer,image_dataset):
    "saves model info"
    model.class_to_idx = image_dataset.class_to_idx
    torch.save({'model':'vgg16',
           'learning_rate':args.learning_rate,
           'input_size':25088,
           'output_size':102,
           'epochs':args.epochs,
           'classifier':model.classifier,
           'state_dict':model.state_dict(),
           'class_to_idx':model.class_to_idx,
           'optimizer':optimizer.state_dict()},args.save_directory)

def restore_model(path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model = getattr(torchvision.models, checkpoint['model'])(pretrained=True)
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.learning_rate = checkpoint['learning_rate']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    return model

    
    
    
    
    
        
    
    