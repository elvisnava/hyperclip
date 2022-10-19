import os

import clip
import numpy as np
import torch
import tqdm
from PIL import Image
from slugify import slugify
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from utils import clip_utils

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

base_path = os.path.dirname(os.path.dirname(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

def _convert_image_to_rgb(image):
            return image.convert("RGB")

def compute_image_features(model):

    print("Load CLIP {} model".format(model))
    clip_model, _ = clip.load(model, device=torch.device("cpu"), jit=False) # load cpu version to get float32 model
    clip_model = clip_model.to(device)
    image_resolution = clip_utils.image_resolution[model]
    preprocess = Compose([Resize((image_resolution,image_resolution), interpolation=BICUBIC),
                            _convert_image_to_rgb,
                            ToTensor(),
                            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                            ])

    img_features = {}

    for data_subtype in ['train2014', 'val2014']:

        img_dir = base_path + f'/data/VQA/Images/{data_subtype}/'

        for filename in tqdm.tqdm(os.listdir(img_dir)):

            img_name = filename.replace(".jpg", "")
            image = preprocess(Image.open(img_dir + filename)).unsqueeze(0).to(device)

            with torch.no_grad():
                features = clip_model.encode_image(image)

            img_features[img_name] = features.float().cpu().numpy()
            
    np.save(base_path+f'/data/VQA/Features/ImageFeatures/image_features_{slugify(model)}.npy',img_features)

def load_image_features(model):
    img_features = np.load(base_path + f"/data/VQA/Features/ImageFeatures/image_features_{slugify(model)}.npy", allow_pickle=True).item()

    for img_name in img_features.keys():
        img_features[img_name] = torch.from_numpy(img_features[img_name]).to(device)

    return img_features
