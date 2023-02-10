import numpy as np
import torch
import json
import argparse
from model_utils import restore_model
from Transforms import process_image
def prediction_system(model,processed_image,cat_class,topk,gpu):
    if gpu == "gpu":
        model.to('cuda')
    else:
        model.to('cpu')
    model.eval() #set model to eval
    img = process_image(processed_image)
    img = img.numpy()
    img = torch.from_numpy(np.array([img]))
    img = img.float()
    #here the tensor is converted into a float numpy array

    with torch.no_grad():
        logps = model.forward(img.cuda() if gpu=='gpu' else img)
        
    prblit = torch.exp(logps).data.topk(topk) #selecting the top 5 probablities
    probs = prblit[0] #getting the probablities
    classes = prblit[1]#labels for classes
    classes_to_idx_reverse = {model.class_to_idx[f]: f for f in model.class_to_idx}#mapping model categories 
    classes_final_indexes = [classes_to_idx_reverse[label] for label in classes.cpu().numpy()[0]]#transferring classes to cpu
    final_names = [cat_class[index] for index in classes_final_indexes]
    final_probs = probs[0].cpu().numpy()
    print("Model predicts the image as :")
    for i in range(topk):
        print(f"{final_names[i]} with a probablity of {final_probs[i]}")
    #note: function returns probablities along with species names
if __name__ == "__main__":
    args = argparse.ArgumentParser(description = 'Parser for prediction system')
    args.add_argument('--input', default='/home/workspace/ImageClassifier/flowers/test/18/image_04272.jpg', type = str)
    args.add_argument('--checkpoint', default='/home/workspace/ImageClassifier/saved_model.pth',type = str)
    args.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    args.add_argument('--category_classes', dest="category_classes", action="store", default='/home/workspace/ImageClassifier/cat_to_name.json')
    args.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    args = args.parse_args()
    
    new_model = restore_model(args.checkpoint)
    with open(args.category_classes,'r') as f:
        c_to_c = json.load(f)
    prediction_system(new_model,args.input,c_to_c,args.top_k,args.gpu)
    
    
    