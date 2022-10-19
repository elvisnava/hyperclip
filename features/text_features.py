import json
import os

import clip
import numpy as np
import torch
import tqdm
from slugify import slugify

base_path = os.path.dirname(os.path.dirname(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_text_features(model):
    print("Load CLIP {} model".format(model))
    clip_model, _ = clip.load(model, device=torch.device("cpu"), jit=False) # load cpu version to get float32 model
    clip_model = clip_model.to(device)

    with open(base_path+"/data/VQA/Meta/ques_ans_count.json") as file:
        qa = json.load(file)

    with open(base_path+"/data/VQA/Annotations/prompt_meta.json") as file:
        prompt_meta = json.load(file)

    text_features_meta = {}
    for ques in tqdm.tqdm(qa.keys()):
        temp = prompt_meta[ques]
        answers = list(qa[ques].keys())
        prompts = [temp.format(a.replace("_", " ")) for a in answers]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        text_features = text_features.float().cpu().numpy()
        
        text_features_meta[ques] = text_features

    # with open(base_path+"/data/VQA/Features/TextFeatures/text_features.json", "w") as outfile:
    #     json.dump(text_features_val, outfile)
    np.save(base_path+f'/data/VQA/Features/TextFeatures/text_features_{slugify(model)}.npy',text_features_meta)

def load_text_features(model):
    text_features = np.load(base_path + f"/data/VQA/Features/TextFeatures/text_features_{slugify(model)}.npy", allow_pickle=True).item()

    for ques in text_features.keys():
        text_features[ques] = torch.from_numpy(text_features[ques]).to(device)

    return text_features
