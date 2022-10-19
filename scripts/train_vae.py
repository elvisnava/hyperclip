# Takes a base model and learns a generative hypernetwork using adapted weights as data.

import argparse
import json
import os
from functools import partial

import numpy as np
import torch
import wandb
from features.image_features import load_image_features
from features.ques_features import load_ques_features
from features.text_features import load_text_features
from model.custom_hnet import HyperEncoder, HyperGenerator
from training.vae_learn import VAETraining
from utils.build_opt import build_optimizer
from utils.init_utils import load_metamodel_from_checkpoint
from utils.misc_utils import str2bool
from utils.train_utils import log_metric

parser = argparse.ArgumentParser()
parser.add_argument('--wandb_mode', type=str, default='online',
                    help='Set to "disabled" to disable Weights & Biases logging')

parser.add_argument('--few_shot_checkpoint', type=str, help='required')
parser.add_argument('--precompute_checkpoint', type=str, help='required')

parser.add_argument('--vae_epochs', type=int)
parser.add_argument('--vae_batch_size', type=int)
parser.add_argument('--vae_noise_dim', type=int)

parser.add_argument('--vae_optimizer', type=str)
parser.add_argument('--vae_learning_rate', type=float)
parser.add_argument('--kld_weight', type=float)

parser.add_argument('--eval_inner_epochs', type=str)
parser.add_argument('--generated_optimizer', type=str)
parser.add_argument('--generated_learning_rate', type=float)

# hyperclip
parser.add_argument('--vae_hidden_dim', type=str)
parser.add_argument('--val_epoch_interval', type=int)
base_path = os.path.dirname(os.path.dirname(__file__))

parser.add_argument('--normalize', type=str2bool)
parser.add_argument('--grad_clip', type=float)
parser.add_argument('--save_checkpoint', type=str2bool)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
torch.manual_seed(args.seed)
rng = np.random.RandomState(args.seed)
np.random.seed(args.seed)
default_config = {
    "inner_epochs": 50,
    "inner_optimizer": "sgd",
    "inner_learning_rate": 0.1,
    "train_subtype": "random",
    "val_subtype": "random",

    "vae_hidden_dim": "128,128",
    "vae_noise_dim": 128,

    "vae_epochs": 10000,
    "vae_batch_size": 32,
    "kld_weight":1,
    "vae_optimizer": "adam",
    "vae_learning_rate": 0.0001,

    "eval_inner_epochs": '50',
    "generated_optimizer": "adam",
    "generated_learning_rate": 0.001,
    "generated_momentum": 0.9,
    "generated_sgd_nesterov": True,

    "val_epoch_interval": 2000,
    "clip_model": "ViT-L/14@336px",
    "save_checkpoint": False,
    "normalize": False,
    "grad_clip": -1,
}

def main(args):
    cfg = default_config
    cfg.update({k: v for (k, v) in vars(args).items() if v is not None})
    print(cfg)
    wandb.init(project="train_vae", entity="srl_ethz", config=cfg,
               mode=args.wandb_mode)
    config = wandb.config

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(base_path + "/data/VQA/Meta/meta_train.json") as file:
        train_data = json.load(file)
    with open(base_path + "/data/VQA/Meta/meta_test.json") as file:
        test_data = json.load(file)

    meta_module, precomputed_latent, precomputed_latent_train_eval, precomputed_latent_val_eval = \
        load_metamodel_from_checkpoint(config, device)

    # pre-computed features
    image_features = load_image_features(config["clip_model"])
    text_features = load_text_features(config["clip_model"])
    ques_emb = load_ques_features(config["clip_model"])
    
    hidden_dims = [int(h) for h in config["vae_hidden_dim"].split(",")]
    hnet_enc = HyperEncoder(meta_module.mnet, e_dim=config["vae_noise_dim"],
                                hidden_dims=hidden_dims, normalize="normalize" in config and config["normalize"]).to(device)
    hidden_dims.reverse()
    hnet_gen = HyperGenerator(meta_module.mnet, e_dim=config["vae_noise_dim"],
                              hidden_dims=hidden_dims, normalize="normalize" in config and config["normalize"]).to(device)

    optimizer_gen = build_optimizer(hnet_gen.parameters(), config, loop="vae")
    optimizer_enc = build_optimizer(hnet_enc.parameters(), config, loop="vae")

    vae_trainer = VAETraining(meta_module, hnet_gen, hnet_enc, optimizer_gen, optimizer_enc, image_features,
                              text_features, ques_emb, config, device)

    inner_optim =partial(build_optimizer, config=config, loop="inner")
    generated_inner_optim =partial(build_optimizer, config=config, loop="generated")

    # Get STD and MEAN
    if precomputed_latent is not None:
        sampled_precomputed_latent = {k:v[0:1].to(device) for (k,v) in precomputed_latent.items()}
    else:
        sampled_precomputed_latent=None

    _, optimized_params_mean, optimized_params_std, _ =  \
        vae_trainer.run_epoch(train_data, config["inner_epochs"], inner_optim, precomputed_latent=sampled_precomputed_latent,
                               batch_size=config["vae_batch_size"],
                               output_mean_std=True, device=device, train_subtype = config["train_subtype"], val_subtype=config["val_subtype"], debug=True)

    optimized_params_mean, optimized_params_std=optimized_params_mean[0], optimized_params_std[0]
    wandb.log({"mean":optimized_params_mean, "std":optimized_params_std, "std_avr": optimized_params_std.mean().item()})

    if config["normalize"]: 
        optimized_params_std=optimized_params_std.clip(optimized_params_std.mean().item(),optimized_params_std.mean().item())
        hnet_gen.set_stats(optimized_params_mean, optimized_params_std)
        hnet_enc.set_stats(optimized_params_mean, optimized_params_std)
        vae_trainer.set_stats(optimized_params_mean, optimized_params_std)

    for meta_epoch in range(config["vae_epochs"]):
        if precomputed_latent is not None:
            curr_sample_epoch = np.random.randint(0, precomputed_latent["clip_embedding"].shape[0])
            curr_sample_batch_perm = np.random.permutation(precomputed_latent["clip_embedding"].shape[1])
            sampled_precomputed_latent = {k:v[curr_sample_epoch:curr_sample_epoch+1, curr_sample_batch_perm].to(device) for (k,v) in precomputed_latent.items()}
        else:
            sampled_precomputed_latent=None

        vae_trainer.run_epoch(train_data, config["inner_epochs"], inner_optim, precomputed_latent=sampled_precomputed_latent,
                               batch_size=config["vae_batch_size"],
                               train=True, device=device, train_subtype = config["train_subtype"], val_subtype=config["val_subtype"], debug=True)

        if (meta_epoch+1) % (config["val_epoch_interval"]//10+1) == 0:
            log_dict = vae_trainer.run_epoch(train_data,  config["inner_epochs"], inner_optim,
                                             precomputed_latent=precomputed_latent_train_eval, device=device, epoch=meta_epoch)
            log_metric(log_dict, "eval_train/")
            log_dict = vae_trainer.run_epoch(test_data,  config["inner_epochs"], inner_optim,
                                             precomputed_latent=precomputed_latent_val_eval, device=device, epoch=meta_epoch)
            log_metric(log_dict, "eval_val/")

            log_dict = vae_trainer.run_epoch(train_data,  config["inner_epochs"], inner_optim, reconstructed=True,
                                             precomputed_latent=precomputed_latent_train_eval, device=device, epoch=meta_epoch)
            log_metric(log_dict, "reconstr_eval_train/")
            log_dict = vae_trainer.run_epoch(test_data,  config["inner_epochs"], inner_optim, reconstructed=True,
                                             precomputed_latent=precomputed_latent_val_eval, device=device, epoch=meta_epoch)
            log_metric(log_dict, "reconstr_eval_val/")

        if meta_epoch % config["val_epoch_interval"] == config["val_epoch_interval"]-1:

            if config["eval_inner_epochs"] != '':
                for idx, inner_epochs in enumerate(config["eval_inner_epochs"].split(",")):
                    log_dict = vae_trainer.run_epoch(train_data,  int(inner_epochs), generated_inner_optim, device=device, generated=True, epoch=meta_epoch)
                    log_metric(log_dict, "generated_eval_train_{}step/".format(inner_epochs))

                    log_dict = vae_trainer.run_epoch(test_data,  int(inner_epochs), generated_inner_optim, device=device, generated=True, epoch=meta_epoch)
                    log_metric(log_dict, "generated_eval_val_{}step/".format(inner_epochs))

            if config["save_checkpoint"]:
                hnet_gen_output_path_checkpoint = base_path + "/evaluation/vae/hnet_gen_" + str(
                    wandb.run.name) + ".pth"
                hnet_enc_output_path_checkpoint = base_path + "/evaluation/vae/hnet_enc_" + str(
                    wandb.run.name) + ".pth"
                torch.save(hnet_gen.state_dict(), hnet_gen_output_path_checkpoint)
                torch.save(hnet_enc.state_dict(), hnet_enc_output_path_checkpoint)
                print(f"Checkpoint for meta-epoch {meta_epoch} saved!")

    hnet_gen_output_path_checkpoint = base_path + "/evaluation/vae/hnet_gen_" + str(
        wandb.run.name) + ".pth"
    hnet_enc_output_path_checkpoint = base_path + "/evaluation/vae/hnet_enc_" + str(
        wandb.run.name) + ".pth"
    torch.save(hnet_gen.state_dict(), hnet_gen_output_path_checkpoint)
    torch.save(hnet_enc.state_dict(), hnet_enc_output_path_checkpoint)

    wandb.finish()


if __name__ == "__main__":
    main(args)
