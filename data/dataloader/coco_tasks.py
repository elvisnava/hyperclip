from __future__ import division, print_function

import json
import os
import random

import clip
import numpy as np
import torch
import tqdm
from features.image_features import load_image_features
from slugify import slugify
from torch.utils.data import Dataset

base_path =  os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_coco_tasks():
    with open(base_path + "/data/VQA/Meta/meta_train.json") as file:
        train_data = json.load(file)
    with open(base_path + "/data/VQA/Meta/meta_test.json") as file:
        test_data = json.load(file)

    data_types = ['val2014', 'train2014']
    coco_data = {}
    for dt in data_types:
        with open(base_path + f"/data/VQA/Annotations/annotations/instances_{dt}.json") as file:
            tmp = json.load(file)
            if coco_data == {}:
                coco_data = tmp
            else:
                coco_data['images'] += tmp['images']
                coco_data['annotations'] += tmp['annotations']
    
    imgs_test = []
    for task in test_data:
        for dataSubType in ["train", "test"]:
            for [image_name, _] in test_data[task][dataSubType]:
                imgs_test.append(image_name)

    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    imgs_id_to_name = {i['id']: i['file_name'] for i in coco_data['images']}

    vanilla_coco_categories = {}
    for ann in tqdm.tqdm(coco_data["annotations"]):
        if ann["area"] > 20000 and not ann["iscrowd"]:
            cat = cat_id_to_name[ann["category_id"]]
            img_name = imgs_id_to_name[ann['image_id']].split(".")[0]
            if img_name not in imgs_test:
                if cat not in vanilla_coco_categories.keys():
                    vanilla_coco_categories[cat] = []
                if img_name not in vanilla_coco_categories[cat]:
                    vanilla_coco_categories[cat].append(img_name)

    np.save(base_path+"/data/Attributes/vanilla_coco_categories.npy", vanilla_coco_categories)


def compute_coco_answer_features(model='ViT-L/14@336px'):
    print("Load CLIP {} model".format(model))
    clip_model, _ = clip.load(model, device=torch.device("cpu"), jit=False) # load cpu version to get float32 model
    clip_model = clip_model.to(device)

    categories = np.load(base_path+"/data/Attributes/vanilla_coco_categories.npy", allow_pickle=True).item()

    categories_features = {}
    for cat in tqdm.tqdm(categories.keys()):
        prompt = f"A picture of a {cat}"
        prompt = clip.tokenize(prompt).to(device)

        with torch.no_grad():
            text_feature = clip_model.encode_text(prompt)
        
        text_feature = text_feature.float().cpu().numpy()
        
        categories_features[cat] = text_feature

    np.save(base_path+f'/data/Attributes/coco_answer_features_{slugify(model)}.npy',categories_features)
    
def load_coco_answer_features(model='ViT-L/14@336px'):
    coco_answer_features = np.load(base_path+f'/data/Attributes/coco_answer_features_{slugify(model)}.npy', allow_pickle=True).item()
    for cat in coco_answer_features.keys():
        coco_answer_features[cat] = torch.from_numpy(coco_answer_features[cat]).to(device)

    return coco_answer_features

def filter_categories(categories, min_size=5):
    filtered_categories = {}
    for cat in categories.keys():
        if len(categories[cat]) >= min_size:
            filtered_categories[cat] = categories[cat]
    return filtered_categories

class COCO_Tasks(Dataset):

    def __init__(self, categories, dataSubType, image_features, coco_answer_features, n_way=5, train_size=5, test_size=5, task_seed=42):
        self.categories = categories
        self.image_features = image_features
        self.coco_answer_features = coco_answer_features
        self.n_way = n_way
        self.train_size = train_size
        self.test_size = test_size
        self.task_seed = task_seed

        random.seed(self.task_seed)
        self.answers = random.sample(list(self.categories.keys()), n_way)
        self.text_features = torch.cat([self.coco_answer_features[answer] for answer in self.answers], dim=0)

        train_data = []
        test_data = []
        for answer in self.answers:
            data_for_answer = random.sample(self.categories[answer], self.train_size+self.test_size)
            for i, image_name in enumerate(data_for_answer):
                if i < self.train_size:
                    train_data.append([image_name, answer])
                else:
                    test_data.append([image_name, answer])
        
        if dataSubType == "train":
            self.data = train_data
        elif dataSubType == "test":
            self.data = test_data
        elif dataSubType == "traintest":
            self.data = train_data + test_data

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        [image_name, answer] = self.data[idx]
        image_features =  self.image_features[image_name]
        text_features = self.text_features


        sample = {'ques_emb': [0.], 'image_features': image_features, 'text_features': text_features, "label": self.answers.index(answer)}

        return sample

if __name__ == "__main__":
    #compute_coco_tasks()
    #compute_coco_answer_features()
    coco_categories = np.load(base_path+"/data/Attributes/vanilla_coco_categories.npy", allow_pickle=True).item()
    coco_answer_features = load_coco_answer_features(model='ViT-L/14@336px')
    image_features = load_image_features(model='ViT-L/14@336px')
    dataset = COCO_Tasks(coco_categories, "train", image_features, coco_answer_features, n_way=5)
    for sample in dataset:
        print(sample)

