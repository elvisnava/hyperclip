from __future__ import division, print_function

import math

import numpy as np
import torch
from torch.utils.data import Dataset


class CLIP_VQA(Dataset):

    def __init__(self, meta_data, dataSubType, task , image_features, text_features, ques_emb, n_shot=None, n_shot_seed=42,
                                        data_idx=None):
        """
        Args:
            meta_data(string): Path to the meta learning data file
            dataSubType(string): train/test/traintest
            task(string): question
            image_features(dict): image features dict loaded from features.image_features
            text_features(dict): text features dict loaded from features.text_features
            ques_emb(dict): question feature dict loaded from features.ques_features
            n_shot(int): number of examples per class
            n_shot_seed(int): random seed for n_shot sampling
        """
        self.dataSubType = dataSubType
        self.task = task
        self.answers = meta_data[self.task]["answers"]
        self.image_features = image_features
        self.text_features = text_features
        self.ques_emb = ques_emb
        if data_idx is not None:
            self.data = meta_data[self.task]["train"] + meta_data[self.task]["test"]
            self.data = [self.data[i] for i in data_idx]
        elif dataSubType in ["train","test"]:
            self.data = meta_data[self.task][self.dataSubType]
        elif dataSubType == "traintest":
            self.data = meta_data[self.task]["train"] + meta_data[self.task]["test"]
        elif dataSubType == "random":
            self.data = meta_data[self.task]["train"] + meta_data[self.task]["test"]
            frac = torch.rand(())/3.+2./3.
            self.data = [self.data[i] for i in np.random.permutation(len(self.data))[:math.ceil(len(self.data)*frac)]]
        elif dataSubType == "random50":
            self.data = meta_data[self.task]["train"] + meta_data[self.task]["test"]
            frac = 0.5
            self.data = [self.data[i] for i in np.random.permutation(len(self.data))[:math.ceil(len(self.data)*frac)]]
        else:
            raise ValueError

        if n_shot is not None and n_shot != "full":
            all_answers = np.array([a for [_,a] in self.data])
            classes_idx = [np.arange(len(self.data))[np.array(all_answers) == self.answers[i]] for i in range(len(self.answers))]
            classes_idx = [np.random.RandomState(seed=n_shot_seed).permutation(i)[:n_shot] for i in classes_idx]
            self.data = [self.data[i] for i in np.concatenate(classes_idx)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        [image_name, answer] = self.data[idx]
        image_features =  self.image_features[image_name]
        text_features = self.text_features[self.task]
        ques_emb = self.ques_emb[self.task]


        sample = {'ques_emb': ques_emb, 'image_features': image_features, 'text_features': text_features, "label": self.answers.index(answer)}

        return sample
