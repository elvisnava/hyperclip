
embedding_size = {

    'RN50': 1024,
    'RN101': 512,
    'RN50x4': 640,
    'RN50x16': 768,
    'RN50x64': 1024,
    'ViT-B/32': 512,
    'ViT-B/16': 512,
    'ViT-L/14': 768,
    'ViT-L/14@336px': 768

}

image_resolution = {

    'RN50': 224,
    'RN101': 224,
    'RN50x4': 288,
    'RN50x16': 384,
    'RN50x64': 448,
    'ViT-B/32': 224,
    'ViT-B/16': 224,
    'ViT-L/14': 224,
    'ViT-L/14@336px': 336

}

cached_location = {

    'RN50': "~/.cache/clip/RN50.pt",
    'RN101': "~/.cache/clip/RN101.pt",
    'RN50x4': "~/.cache/clip/RN50x4.pt",
    'RN50x16': "~/.cache/clip/RN50x16.pt",
    'RN50x64': "~/.cache/clip/RN50x64.pt",
    'ViT-B/32': "~/.cache/clip/ViT-B-32.pt",
    'ViT-B/16': "~/.cache/clip/ViT-B-16.pt",
    'ViT-L/14': "~/.cache/clip/ViT-L-14.pt",
    'ViT-L/14@336px': "~/.cache/clip/ViT-L-14-336px.pt"

}