# Takes a hypernetwork and learns the hyperclip model on weight generated by embedding adaptation

import argparse
import json
import os
from functools import partial

import numpy as np
import torch
import wandb
from data.dataloader.coco_tasks import (filter_categories,
                                        load_coco_answer_features)
from features.image_features import load_image_features
from features.ques_features import load_ques_features
from features.text_features import load_text_features
from training.store_few_shot_latent import StoreFewShotLatent
from utils.build_opt import build_optimizer
from utils.init_utils import load_metamodel_from_checkpoint
from utils.misc_utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--wandb_mode', type=str, default='online',
                    help='Set to "disabled" to disable Weights & Biases logging')

parser.add_argument('--epochs', type=int)

parser.add_argument('--inner_epochs', type=str)
parser.add_argument('--inner_optimizer', type=str)
parser.add_argument('--inner_learning_rate', type=float)

parser.add_argument('--compute_hessian', type=str2bool)

parser.add_argument('--few_shot_checkpoint', type=str, help='required')
parser.add_argument('--vae_checkpoint', type=str, help='required')
parser.add_argument('--vae_stochastic_init', type=str2bool)

parser.add_argument('--use_extended_coco', type=str2bool)
parser.add_argument('--extend_coco_size', type=int)
parser.add_argument('--extend_coco_frac_train', type=float)

parser.add_argument('--data_subtype', type=str)


base_path = os.path.dirname(os.path.dirname(__file__))

default_config = {
    "epochs": 100,
    "inner_epochs": '10',
    "inner_optimizer": "sgd",
    "inner_learning_rate": 0.1,
    "compute_hessian": False,
    "vae_stochastic_init": False,

    "clip_model": "ViT-L/14@336px",

    "use_extended_coco": False,
    "extend_coco_size": 10 * 870,
    "extend_coco_frac_train": 0.5,
    "data_subtype": "random",
}

torch.manual_seed(42)
rng = np.random.RandomState(42)
np.random.seed(42)

def main(args):
    cfg = default_config
    cfg.update({k: v for (k, v) in vars(args).items() if v is not None})
    print(cfg)
    wandb.init(project='precompute_adaptation', entity="srl_ethz", config=cfg,
               mode=args.wandb_mode)
    config = wandb.config

    log_file = base_path + "/evaluation/precompute_adaptation/" + str(wandb.run.name) + ".pth"
    log_file_train_eval = base_path + "/evaluation/precompute_adaptation/" + str(wandb.run.name) + "_train_eval.pth"
    log_file_val_eval = base_path + "/evaluation/precompute_adaptation/" + str(wandb.run.name) + "_val_eval.pth"

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(base_path + "/data/VQA/Meta/meta_train.json") as file:
        train_data = json.load(file)
    with open(base_path + "/data/VQA/Meta/meta_test.json") as file:
        test_data = json.load(file)

    reset_normal_embedding = config["vae_stochastic_init"] and "vae_checkpoint" in config

    meta_module, _, _, _ = load_metamodel_from_checkpoint(config, device)

    # pre-computed features
    image_features = load_image_features(config["clip_model"])
    text_features = load_text_features(config["clip_model"])
    ques_emb = load_ques_features(config["clip_model"])

    coco_categories = None
    coco_answer_features = None
    if config["use_extended_coco"]:
        coco_categories = np.load(base_path+"/data/Attributes/vanilla_coco_categories.npy", allow_pickle=True).item()
        coco_categories = filter_categories(coco_categories, 10)
        # right now it's hardcoded, it's train_size + test_size for the extended coco sampled datasets
        coco_answer_features = load_coco_answer_features(config["clip_model"])

    unguided_inner_optim = partial(build_optimizer, config=config, loop="inner")
    few_shot_saver = StoreFewShotLatent( meta_module,
                                         image_features=image_features,
                                         text_features=text_features,
                                         ques_emb=ques_emb,
                                         config=config,
                                         device=device, compute_hessian=config["compute_hessian"],
                                         reset_normal_embedding=reset_normal_embedding,
                                         coco_categories=coco_categories, coco_answer_features=coco_answer_features,
                                         extend_coco_size=config["extend_coco_size"])

    print("Computing metric")
    from utils.train_utils import log_metric
    log_dict = few_shot_saver.run_epoch(train_data, config["inner_epochs"], unguided_inner_optim,
                                        batch_size=len(list(train_data.keys())),train=True,
                                        log_file=log_file_train_eval)
    log_metric(log_dict, "eval_train/")
    log_dict = few_shot_saver.run_epoch(test_data, config["inner_epochs"], unguided_inner_optim,
                                        batch_size=len(list(test_data.keys())),train=True,
                                        log_file=log_file_val_eval)
    log_metric(log_dict, "eval_val/")

    for meta_epoch in range(config["epochs"]):
        few_shot_saver.run_epoch(train_data, config["inner_epochs"], unguided_inner_optim,
                                 batch_size=len(list(train_data.keys())),
                                 extend_coco=config["use_extended_coco"],
                                 extend_coco_frac_train=config["extend_coco_frac_train"],
                                 train=True, train_subtype = config["data_subtype"], val_subtype=config["data_subtype"], debug=True, log_file=log_file)


    wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
