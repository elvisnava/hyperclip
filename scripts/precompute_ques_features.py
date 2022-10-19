import argparse

from features.ques_features import compute_ques_features

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ViT-L/14@336px', help='CLIP visual encoder')


if __name__ == "__main__":
    args = parser.parse_args()
    compute_ques_features(args.model)
