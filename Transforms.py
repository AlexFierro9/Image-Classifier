from torchvision import datasets,transforms
import torch
from PIL import Image

def transform_data(data_dir):
    """Returns transformations and datasets for training and validation and test datasets"""
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
                  'train':transforms.Compose([transforms.RandomResizedCrop(size = 224),
                                             transforms.RandomRotation(60),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                  'valid': transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
                  'test': transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
                  }


    image_datasets = {
        'train': datasets.ImageFolder(train_dir,transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir,transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir,transform=data_transforms['test'])
        }


    dataloaders = [torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                    torch.utils.data.DataLoader(image_datasets['valid'], batch_size =64,shuffle = True),
                    torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64, shuffle = True)]
    
    
    return dataloaders[0],dataloaders[1],dataloaders[2],image_datasets['train']

def process_image(path):
    """Scales and transforms an image for use with prediction system"""
    img_pil = Image.open(path)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)
    #note: returns a tensor, not a numpy array
    
    return image