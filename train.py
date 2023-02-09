import numpy as np
from torch import nn,optim
from torchvision import datasets, models, transforms
import torch.utils.data 
import pandas as pd
import matplotlib.pyplot as plt
import random

from PIL import Image
import time
import os
import argparse 
from Transforms import transform_data
from model_utils import network_setup_systems, save_model_func
import torch


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest = 'data_directory',help = "Training samples address",default = '/flowers')
    
    #optional
    parser.add_argument('--model_arch', dest='model_arch', help="Model backend", default="vgg16", type=str, choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--epochs', dest='epochs', help="Number of times you want to train", default=5, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', help="Learning rate, give a value b/w 0 & 1, default set to 0.001", default=0.001, type=float)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=2048)
    parser.add_argument('--save_directory', dest='save_directory', help="Address where model will be saved, default value already set", default='/home/workspace/ImageClassifier/saved_model.pth')
    parser.add_argument('--gpu', dest='gpu', help="Recommended to use gpu for training purposes", action='store_true')
    source_args = parser.parse_args()
    trainload,validload,testload,train_data = transform_data(source_args.data_directory)
    model , criterion,optimizer = network_setup_systems(source_args)
    print("Training Started")
    print_every  = 20
    steps = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    


    for epochs in range(source_args.epochs):
        running_loss = 0
        if torch.cuda.is_available():
            model.cuda()
        model.train()
        for images,labels in trainload:
            steps+=1
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps%print_every==0:
                test_loss  = 0 
                accuracy = 0 
                model.eval()
                with torch.no_grad():
                    for images,labels in validload:
                        images,labels = images.to('cuda'),labels.to('cuda')
                        output = model.forward(images)
                        loss = criterion(output,labels)
                        test_loss += loss.item()
                        ps = torch.exp(output)
                        top_p,top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                    print(f"Epoch: {epochs+1}/{source_args.epochs} Tra_loss = {running_loss/print_every}Val_loss ={test_loss/len(validload)} Val_acc = {accuracy/len(validload)}")
                running_loss = 0
                model.train()
               
    print("Training Complete\nSaving Model")
    
        
    save_model_func(source_args,model,optimizer,train_data) 
    print("Model Saved")
    

    
    
    
    
        
    
