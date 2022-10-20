import argparse
from functools import partial
import json
import os
from copy import deepcopy

import model.custom_hnet as custom_hnet
import numpy as np
import torch
import wandb
from data.dataloader.clip_vqa import CLIP_VQA
from features.image_features import load_image_features
from features.ques_features import load_ques_features
from features.text_features import load_text_features
from model.hyperclip import build_hyperclip_from_classic_clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.store_few_shot_latent import StoreFewShotLatent
from training.vae_learn import sample_from_enc
from utils import clip_utils
from utils.build_opt import build_optimizer
from utils.init_utils import load_metamodel_from_checkpoint, load_vae_and_metamodel_from_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--wandb_mode', type=str, default='online', help='Set to "disabled" to disable Weights & Biases logging')
parser.add_argument('--run_id', type=str, default='srl_ethz/hyperclip-scripts/irwxxdeh', help='The full "<entity>/<project>/<run_id>" identifier of the run to load')
parser.add_argument('--task_batch_size', type=int, default=10, help='Size of randomly sampled tasks for task classification')

base_path = os.path.dirname(os.path.dirname(__file__))

default_config = {
    "inner_epochs": '50',
    "inner_optimizer": "sgd",
    "inner_learning_rate": 0.1,

    "vae_stochastic_init": False,
}

torch.manual_seed(42)
rng = np.random.RandomState(42)
np.random.seed(42)

def main(args):

    cfg = default_config
    cfg.update({k: v for (k, v) in vars(args).items() if v is not None})
    api = wandb.Api()
    loaded_run = api.run(args.run_id)
    cfg.update({k: v for (k, v) in loaded_run.config.items() if v is not None and k not in cfg})
    wandb.init(project="hyperclip-classification", entity="srl_ethz", name=loaded_run.name, config=cfg, mode=args.wandb_mode)
    config = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(base_path + "/data/VQA/Meta/meta_train.json") as file:
        train_data = json.load(file) 
    with open(base_path + "/data/VQA/Meta/meta_test.json") as file:
        test_data = json.load(file)

    train_tasks = list(train_data.keys())
    test_tasks = list(test_data.keys())

    hyperclip_path = base_path + "/evaluation/hyperclip/hyperclip_"+str(loaded_run.name)+".pth"
    
    hnet_gen, hnet_enc = None, None
    if "vae_checkpoint" in config and config["vae_checkpoint"] is not None:
        meta_module, hnet_gen, hnet_enc, precomputed_latent, precomputed_latent_train_eval, precomputed_latent_val_eval = \
            load_vae_and_metamodel_from_checkpoint(config, device)
    else:
        meta_module, precomputed_latent, precomputed_latent_train_eval, precomputed_latent_val_eval = \
            load_metamodel_from_checkpoint(config, device)

    # pre-computed features
    image_features = load_image_features(config["clip_model"])
    text_features = load_text_features(config["clip_model"])
    ques_emb = load_ques_features(config["clip_model"])

    hyperclip = build_hyperclip_from_classic_clip(
        os.path.expanduser(clip_utils.cached_location[config["clip_model"]]),
        hyper_model=config["hyperclip_model"],
        mainnet_param_count=meta_module.mnet.get_parameter_vector().shape[0],
        hyper_hidden_dims=[] if config["hyperclip_hidden_dim"] == "" else [int(i) for i in config["hyperclip_hidden_dim"].split(",")],
        pretrained_it_location=os.path.expanduser(clip_utils.cached_location[config["clip_model"]]),
        pretrained_hyper_location=None).to(device)

    hyperclip.hyper.load_state_dict(torch.load(hyperclip_path), strict=False)
    hyperclip.hyper.eval()

    unguided_inner_optim = partial(build_optimizer, config=config, loop="inner")
    few_shot_saver = StoreFewShotLatent( meta_module,
                                         image_features=image_features,
                                         text_features=text_features,
                                         ques_emb=ques_emb,
                                         config=config,
                                         device=device, compute_hessian=False,
                                         reset_normal_embedding=config["vae_stochastic_init"] and "vae_checkpoint" in config)

    _, _, _, train_tasks_optimized_params = few_shot_saver.run_epoch(train_data, config["inner_epochs"], unguided_inner_optim,
                                            batch_size=len(list(train_data.keys())),
                                            train=False, skip_cond=True,
                                            train_subtype = config["train_subtype"], val_subtype=config["val_subtype"],
                                            output_mean_std=True,
                                            debug=True)


    hyperclip_ques_batch = torch.zeros(config["task_batch_size"], clip_utils.embedding_size[config["clip_model"]], dtype=torch.float32).to(device)
    hyperclip_net_batch = torch.zeros(config["task_batch_size"], hyperclip.hyper.input_dim, dtype=torch.float32).to(device)

    train_correct = 0
    tot_loss_train = 0

    for i, task in enumerate(tqdm(train_tasks)):
        
        prob_other_sample = np.ones(len(train_tasks)) * (1.0 / (len(train_tasks)-1))
        prob_other_sample[i] = 0.0
        other_task_ids = np.random.choice(len(train_tasks), size=config["task_batch_size"]-1, replace=False, p=prob_other_sample)

        hyperclip_ques_batch[0] = ques_emb[task]

        opt_latent = train_tasks_optimized_params[i]
        opt_weights = get_weights(meta_module, hnet_gen, hnet_enc, opt_latent, config)
        hyperclip_net_batch[0] = opt_weights
        
        for batch_i, other_id in enumerate(other_task_ids):
            hyperclip_ques_batch[batch_i+1] = ques_emb[train_tasks[other_id]]
            opt_latent = train_tasks_optimized_params[other_id]
            opt_weights = get_weights(meta_module, hnet_gen, hnet_enc, opt_latent, config)
            hyperclip_net_batch[batch_i+1] = opt_weights

        with torch.no_grad():
            (features, logits), hyperclip_loss = hyperclip.forward(hyper = hyperclip_net_batch, text = hyperclip_ques_batch, precomputed_it_embs = True)
            tot_loss_train += hyperclip_loss.detach().cpu().numpy()

            _, logits_hyper_ques, _ = logits

            logits_hyper_ques = logits_hyper_ques[0] #consider the classification of the true task at position 0 against randomly sampled other tasks
            _, pred_index = logits_hyper_ques.topk(1)
            if pred_index == 0:
                train_correct += 1
    
    train_accuracy = train_correct / len(train_tasks)
    print(f"Trainset hyperclip accuracy: {train_accuracy}")
    train_avg_loss = tot_loss_train / len(train_tasks)
    print(f"Trainset hyperclip average loss: {train_avg_loss}")
    wandb.log({"hyperclip_trainset_accuracy": train_accuracy, "hyperclip_trainset_avg_loss": train_avg_loss})


    _, _, _, test_tasks_optimized_params = few_shot_saver.run_epoch(test_data, config["inner_epochs"], unguided_inner_optim,
                                            batch_size=len(list(test_data.keys())),
                                            train=False, skip_cond=True,
                                            train_subtype = config["train_subtype"], val_subtype=config["train_subtype"],
                                            output_mean_std=True,
                                            debug=True)

    test_correct = 0
    tot_loss_test = 0

    for i, task in enumerate(tqdm(test_tasks)):
        
        prob_other_sample = np.ones(len(test_tasks)) * (1.0 / (len(test_tasks)-1))
        prob_other_sample[i] = 0.0
        other_task_ids = np.random.choice(len(test_tasks), size=config["task_batch_size"]-1, replace=False, p=prob_other_sample)

        hyperclip_ques_batch[0] = ques_emb[task]
        opt_latent = test_tasks_optimized_params[i]
        opt_weights = get_weights(meta_module, hnet_gen, hnet_enc, opt_latent, config)
        hyperclip_net_batch[0] = opt_weights
        for batch_i, other_id in enumerate(other_task_ids):
            hyperclip_ques_batch[batch_i+1] = ques_emb[test_tasks[other_id]]
            opt_latent = test_tasks_optimized_params[other_id]
            opt_weights = get_weights(meta_module, hnet_gen, hnet_enc, opt_latent, config)
            hyperclip_net_batch[batch_i+1] = opt_weights

        with torch.no_grad():
            (features, logits), hyperclip_loss = hyperclip.forward(hyper = hyperclip_net_batch, text = hyperclip_ques_batch, precomputed_it_embs = True)
            tot_loss_test += hyperclip_loss.detach().cpu().numpy()

            _, logits_hyper_ques, _ = logits

            logits_hyper_ques = logits_hyper_ques[0] #consider the classification of the true task at position 0 against randomly sampled other tasks
            _, pred_index = logits_hyper_ques.topk(1)
            if pred_index == 0:
                test_correct += 1
    
    test_accuracy = test_correct / len(test_tasks)
    print(f"Testset hyperclip accuracy: {test_accuracy}")
    test_avg_loss = tot_loss_test / len(test_tasks)
    print(f"Testset hyperclip average loss: {test_avg_loss}")
    wandb.log({"hyperclip_testset_accuracy": test_accuracy, "hyperclip_testset_avg_loss": test_avg_loss})

    wandb.finish()
    

def get_weights(meta_module, hnet_gen, hnet_enc, opt_latent, config):
    if "vae_checkpoint" in config and config["vae_checkpoint"] is not None and hnet_gen is not None:
        return hnet_gen(sample_from_enc(hnet_enc, get_weights_from_metamodule(meta_module, opt_latent))).detach()
    return get_weights_from_metamodule(meta_module, opt_latent)

def get_weights_from_metamodule(meta_module, opt_latent):
    inner_params = {k: v.clone().detach().requires_grad_() for (k,v) in meta_module.get_inner_params().items()}
    inner_params["enet.embedding"] = opt_latent.clone().detach().requires_grad_()
    return meta_module.get_mainnet_weights(params = inner_params).detach()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
