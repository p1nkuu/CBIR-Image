from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
import cv2 
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import torch
from torchvision import transforms
from huggingface_hub import login, hf_hub_download
import numpy as np 
import faiss
import json
from random import sample


login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# pretrained=True needed to load UNI weights (and download weights for the first time)
# init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True) #Create Function
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
UNI_model = (model, transform)

def generate_embeddings_w_UID(caption_file, model=UNI_model[0], subset=False):
    """
    Generates and saves image embeddings using pretrained UNI model and 
    assigns UID from caption (.json)file to embeddings

    input: 
        filenames: list of String filenames 
        caption_file: .json file of captions, primarily for uid retrieval

    output: 
        None
    """


    with open(caption_file, 'r') as f:
        data = json.load(f)

    feature_embeddings = []
    uid_to_embeddings_map = {}
    i = 0

    if subset == True:
        sample_size = input('How large of a subset would you like?')
        while i <= int(sample_size):
            for key, item in data.items():
                img_str = item['uuid']#Image filename as String
                uid = key

                image = convert_to_RGB(img_str+'.png')

                with torch.inference_mode():
                    feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,1024]

                uid_to_embeddings_map[uid] = feature_emb
                i+=1
    else:
        for key, item in data.items():
                img_str = item['uuid']#Image filename as String
                uid = key

                image = convert_to_RGB(img_str+'.png')

                with torch.inference_mode():
                    feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,1024]

                uid_to_embeddings_map[uid] = feature_emb

    uid_to_embeddings_map = np.array(feature_embeddings)
    save_embeddings(uid_to_embeddings_map)

def generate_embeddings(filenames, model=UNI_model[0]):
    """
    Generates and saves image embeddings using pretrained UNI model

    input: 
        filenames: list of String filenames 

    output: 
        None
    """
    
    feature_embeddings = []

    #Need to convert from RGBA to RGB then transform
    for img_str in filenames:
        image = convert_to_RGB(img_str)

        #extract features 
        with torch.inference_mode():
            feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,1024]
            feature_embeddings.append(feature_emb) 

    feature_embeddings = np.array(feature_embeddings)

    save_embeddings(feature_embeddings)

def save_embeddings(embeddings_list):
    """
    Saves embeddings as .npy file

    input: 
        embeddings_list: list of embeddings generated from model

    output:
        None

    """
    save_name = input( "Please name your embeddings .npy file (No '.npy' necessary. Example input: books_set_embeddings)")
    np.save(save_name, embeddings_list)


def convert_to_RGB(file_str,transform=UNI_model[1]):
        """
        Converts images to RGB colorspace
        
        input:
            file_str (String filename of image)
            
        output:
            Transformed image
        """

        image = Image.open(file_str) #Create function
        image_rgb = image.convert('RGB')
        image = transform(image_rgb).unsqueeze(dim=0) # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
        return image


def shrink_dataset_random(filenames, sample_size=50):
    """
    Returns random subset of data given sample size
    
    input:
        filenames: list of string filenames
        sample_size: int representing how large desired dataset is

    output:
        list of string filenames 
    """
    return sample(filenames, sample_size)

def img_folder_to_str(folder_path, filetypes):
    """
    Converts folder of images to list of string filenames
    
    input:
        folder_path: string of folder path on local machine
        filetypes: list of strings of filetypes in folder (eg. ['.png', '.jpg'])
    output:
        list of string filenames
    """
    filename_strings =[]
    folder_path_object = []
    for types in filetypes:
        folder_path_object.append( Path(folder_path).glob("*"+types) )
    
    for path_obj in folder_path_object:
        filename_strings = filename_strings + [str(p) for p in path_obj]

    return filename_strings

def create_index(d, embeddings_to_add):
    """
    create FAISS index object
    
    input: 
        d: dimension of embeddings
        embeddings_to_add: list of embeddings you want to add to index object. 

    output:
        Index object"""
    index = faiss.IndexFlatL2(d) 

    for files in embeddings_to_add:
        index.add(files)
    
    return index

def perform_knn(k, index, query_img):
    """
    perform KNN using index and query image
    
    input:
        k: int representing how many nearest neighbors you want to retrieve
        index: FAISS index object with embeddings (should be dataset + evaluation set)
        query_img: query image embedding
        
    output:
        tuple: (D = distance metric, I = indices of k nearest neighbors in Index object"""
    k = 4                         # we want to see 4 nearest neighbors
    D, I = index.search(query_img, k) # sanity check

    D = D[0].tolist()
    I = I[0].tolist()

    return (D, I)

def KNN_filenames(I,filenames):
    """
    Get filenames from KNN evaluation. Can use for visualization (using Image)

    input: 
        I: from perform_knn
        filenames: Same list of embeddings added to index used in create_index
    
    output:
        list of corresponding filenames retrieved by KNN
    """
    results = [filenames[I[0]], filenames[I[1]], filenames[I[2]], filenames[I[3]]]
    return results

def convert_3D_to_2D(embeddings):
    """
    Converts 3D embeddings to 2D embeddings so they are compatible with FAISS
    
    input: 
        embeddings: .npy file of embeddings
    output:
        embeddings loaded from .npy file but 2D
    """
    data = np.load(embeddings) #Outputs 3D embeddings so..
    new_data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
    return new_data

generate_embeddings_w_UID('/Users/genesishang/bttai_novartis/books_set/captions.json', subset=50)