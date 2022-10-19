import json
import os

import clip
import numpy as np
import torch
import tqdm
from slugify import slugify

base_path = os.path.dirname(os.path.dirname(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_ques_features(model):
    print("Load CLIP {} model".format(model))
    clip_model, _ = clip.load(model, device=torch.device("cpu"), jit=False) # load cpu version to get float32 model
    clip_model = clip_model.to(device)

    with open(base_path+"/data/VQA/Meta/ques_ans_count.json") as file:
        qa = json.load(file)

    ques_features = {}
    for ques in tqdm.tqdm(qa.keys()):
        text = clip.tokenize(ques).to(device)

        with torch.no_grad():
            features = clip_model.encode_text(text)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        features =  features.float().cpu().numpy() #np.squeeze(features.float().cpu().numpy())
        
        ques_features[ques] = features

    # with open(base_path+"/data/VQA/Features/TextFeatures/text_features.json", "w") as outfile:
    #     json.dump(text_features_val, outfile)
    np.save(base_path+f'/data/VQA/Features/QuesFeatures/ques_features_{slugify(model)}.npy',ques_features)

def load_ques_features(model):
    ques_features = np.load(base_path + f"/data/VQA/Features/QuesFeatures/ques_features_{slugify(model)}.npy", allow_pickle=True).item()

    for ques in ques_features.keys():
        ques_features[ques] = torch.from_numpy(ques_features[ques]).to(device)

    return ques_features
