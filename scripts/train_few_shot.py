import argparse
import json
import os

import numpy as np
import torch
import wandb
from data.dataloader.coco_tasks import (filter_categories,
                                        load_coco_answer_features)
from features.image_features import load_image_features
from features.ques_features import load_ques_features
from features.text_features import load_text_features
from model.custom_hnet import MetaModel
from training.maml_learn import MAML
from utils import clip_utils
from utils.build_opt import build_optimizer
from utils.misc_utils import str2bool
from utils.train_utils import log_metric, n_shot_trials_run

parser = argparse.ArgumentParser()
parser.add_argument('--wandb_mode', type=str, default='online',
                    help='Set to "disabled" to disable Weights & Biases logging')

parser.add_argument('--meta_epochs', type=int)
parser.add_argument('--meta_batch_size', type=int)

parser.add_argument('--inner_epochs', type=str)

parser.add_argument('--inner_learning_rate', type=float)
parser.add_argument('--eval_inner_epochs', type=str)
parser.add_argument('--second_order', type=str2bool)
parser.add_argument('--train_subtype', type=str)
parser.add_argument('--val_subtype', type=str)
parser.add_argument('--meta_optimizer', type=str)
parser.add_argument('--meta_learning_rate', type=float)
parser.add_argument('--meta_grad_clip', type=float)

parser.add_argument('--alpha', type=float)

# meta_module
parser.add_argument('--inner_param', type=str)
parser.add_argument('--hypernet_hidden_dim', type=str)
parser.add_argument('--straight_through', type=str2bool)
parser.add_argument('--embedding_dim', type=int)

parser.add_argument('--keep_tasks_frac', type=float)

parser.add_argument('--load_checkpoint', type=str)
parser.add_argument('--val_epoch_interval', type=int)

parser.add_argument('--use_extended_coco', type=str2bool)
parser.add_argument('--extend_coco_size', type=int)
parser.add_argument('--extend_coco_frac_train', type=float)

parser.add_argument('--use_clip_embedding_init', type=str2bool)
parser.add_argument('--save_checkpoint', type=str2bool)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--eval', type=str2bool, default=False)

parser.add_argument('--n_shot_trials_maxN', type=int)

base_path = os.path.dirname(os.path.dirname(__file__))

args = parser.parse_args()

torch.manual_seed(args.seed)
rng = np.random.RandomState(args.seed)
np.random.seed(args.seed)

default_config = {
    # meta_module
    "use_clip_embedding_init": False,
    "inner_param": "enet",
    "hypernet_hidden_dim": "128,128,128",
    "straight_through": False,
    "mainnet_use_bias": True,
    "mainnet_hidden_dim": [256],
    "embedding_dim": 128,

    "alpha": 1,

    "clip_model": "ViT-L/14@336px",

    "inner_epochs": "10",
    "inner_learning_rate": 0.1,
    "train_subtype": "test",
    "val_subtype": "train",

    "meta_epochs": 1000,
    "meta_batch_size": 32,
    "second_order": False,
    "meta_grad_clip": 10,

    "eval_inner_epochs": '',
    "meta_optimizer": "adam",
    "meta_learning_rate": 0.001,

    "val_epoch_interval": 25,
    "save_checkpoint": False,
    "load_checkpoint": "",
    "keep_tasks_frac": 1,

    "use_extended_coco": False,
    "extend_coco_size": 10 * 870,
    "extend_coco_frac_train": 0.5
}


def main():
    cfg=default_config
    cfg.update({k:v for (k,v) in vars(args).items() if v is not None})

    wandb.init(project="train_few_shot", entity="srl_ethz", config=cfg,
               mode=args.wandb_mode)
    config = wandb.config

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(base_path + "/data/VQA/Meta/meta_train.json") as file:
        train_data = json.load(file)
    with open(base_path + "/data/VQA/Meta/meta_test.json") as file:
        test_data = json.load(file)

    # pre-computed features
    image_features = load_image_features(config["clip_model"])
    text_features = load_text_features(config["clip_model"])
    ques_emb = load_ques_features(config["clip_model"])

    coco_categories = None
    coco_answer_features = None
    if config["use_extended_coco"]:
        coco_categories = np.load(base_path+"/data/Attributes/vanilla_coco_categories.npy", allow_pickle=True).item()
        coco_categories = filter_categories(coco_categories, 10) # right now it's hardcoded, it's train_size + test_size for the extended coco sampled datasets
        coco_answer_features = load_coco_answer_features(config["clip_model"])

    meta_module = MetaModel(
       inner_param=config["inner_param"],
       mainnet_use_bias=config["mainnet_use_bias"],
       mainnet_hidden_dim=config["mainnet_hidden_dim"],
       hypernet_hidden_dim=[] if config["hypernet_hidden_dim"]=="" else [int(i) for i in config["hypernet_hidden_dim"].split(",")],
       embedding_dim=config["embedding_dim"] if not config["use_clip_embedding_init"] else clip_utils.embedding_size[config["clip_model"]],
       straight_through=config["straight_through"],
       config=config).to(device)

    if "checkpoint" in config:
        loaded_model_path = config["checkpoint"]
        meta_module.load_state_dict(torch.load(loaded_model_path), strict=False)

    meta_optimizer = build_optimizer(meta_module.meta_params, config, loop="meta")
    meta_trainer = MAML(meta_module, meta_optimizer, image_features, text_features, ques_emb, config, coco_categories, coco_answer_features, extend_coco_size=config["extend_coco_size"])

    if "n_shot_trials_maxN" in config and config["n_shot_trials_maxN"] is not None:
        n_shot_trials = []

    best_val_acc=[0]*(1+len(config["eval_inner_epochs"].split(",")))
    best_val_epoch=[0]*(1+len(config["eval_inner_epochs"].split(",")))

    for meta_epoch in range(config["meta_epochs"]):

        if not config["eval"]:
            meta_trainer.run_epoch(train_data, config["inner_epochs"], config["inner_learning_rate"], meta_batch_size=config["meta_batch_size"],
                               train=True, second_order=config["second_order"],
                               train_subtype = config["train_subtype"], val_subtype=config["val_subtype"], debug=True, keep_tasks_frac=config["keep_tasks_frac"],
                               extend_coco=config["use_extended_coco"], extend_coco_frac_train=config["extend_coco_frac_train"], device=device)

        if config["eval"] or meta_epoch % config["val_epoch_interval"] == 0 or meta_epoch == config["meta_epochs"]-1:
            log_dict = meta_trainer.run_epoch(train_data,  config["inner_epochs"], config["inner_learning_rate"], keep_tasks_frac=config["keep_tasks_frac"], device=device, epoch=meta_epoch)
            log_metric(log_dict, "eval_train/")
            log_dict = meta_trainer.run_epoch(test_data,  config["inner_epochs"], config["inner_learning_rate"], device=device, epoch=meta_epoch)

            if best_val_acc[0] < log_dict["query_accuracy_end"]:
                best_val_acc[0] = log_dict["query_accuracy_end"]
                best_val_epoch[0] = meta_epoch
            log_dict["best_accuracy"] = best_val_acc[0]
            log_dict["best_epoch"] = best_val_epoch[0]
            log_metric(log_dict, "eval_val/")

            if log_dict["query_accuracy_end"] < 0.3:
                print("Stopping training")
                return

            if "n_shot_trials_maxN" in config and config["n_shot_trials_maxN"] is not None:
                n_shot_trial_dict = {"epoch": meta_epoch}

            if config["eval_inner_epochs"] != '':
                for idx, inner_epochs in enumerate(config["eval_inner_epochs"].split(",")):
#                    log_dict = meta_trainer.run_epoch(train_data,  int(inner_epochs), config["inner_learning_rate"], keep_tasks_frac=config["keep_tasks_frac"], device=device)
#                    log_dict["epoch"]=meta_epoch
#                    log_metric(log_dict, "eval_train_{}step/".format(inner_epochs))

                    log_dict = meta_trainer.run_epoch(test_data,  int(inner_epochs), config["inner_learning_rate"], device=device, epoch=meta_epoch)

                    if best_val_acc[idx+1] < log_dict["query_accuracy_end"]:
                        best_val_acc[idx+1] = log_dict["query_accuracy_end"]
                        best_val_epoch[idx+1] = meta_epoch
                    log_dict["best_accuracy"] = best_val_acc[idx+1]
                    log_dict["best_epoch"] = best_val_epoch[idx+1]

                    log_metric(log_dict, "eval_val_{}step/".format(inner_epochs))

                    n_shot_trials_run(meta_trainer, n_shot_trial_dict, config, f"eval_val_{inner_epochs}step/", test_data, int(inner_epochs), config["inner_learning_rate"], device=device, epoch=meta_epoch)

            if "n_shot_trials_maxN" in config and config["n_shot_trials_maxN"] is not None:
                n_shot_trials += [n_shot_trial_dict]

        if config["eval"]:
            return

        if config["save_checkpoint"] and (meta_epoch+1) % 20 == 0:
            model_output_path_checkpoint = base_path + "/evaluation/few_shot/meta_module" + str(wandb.run.name) + "_" + str( meta_epoch) + ".pth"
            torch.save(meta_module.state_dict(), model_output_path_checkpoint)
            print(f"Checkpoint for meta-epoch {meta_epoch} saved!")

        if "n_shot_trials_maxN" in config and config["n_shot_trials_maxN"] is not None:
            model_output_path_checkpoint = base_path + "/evaluation/few_shot/few_shot_" + str(
                    wandb.run.name) + "_n_shot.npy"
            np.save(model_output_path_checkpoint, n_shot_trials)

    model_output_path_checkpoint = base_path + "/evaluation/few_shot/few_shot_" + str(wandb.run.name) + ".pth"
    torch.save(meta_module.state_dict(), model_output_path_checkpoint)

    wandb.finish()


if __name__ == "__main__":
    main()
