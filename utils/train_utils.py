import torch
import wandb
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
import copy
from utils.misc_utils import append_dict
import numpy as np

def get_pred( meta_module, dataloader, params=None, embedding=None):
    y_pred = []
    y_true = []
    for sample in dataloader:
        sample_image_features = sample["image_features"]
        sample_text_features = sample["text_features"]
        if embedding is None:
            embedding = sample["ques_emb"][0]
        labels = sample["label"].to(sample_image_features.device)
        similarity = meta_module(sample_image_features, sample_text_features, embedding, params=params)
        y_pred.append(similarity)
        y_true.append(labels)

    return torch.cat(y_pred), torch.cat(y_true)

def test_accuracy( meta_module, dataloader, params=None, embedding=None, params_list=None):
    # Validation inner-loop testing
    meta_module.eval()

    with torch.no_grad():
        if params_list is not None:
            output_list = []
            for p in params_list:
                output, y_true = get_pred(meta_module, dataloader, params=p, embedding=embedding)
                output_list.append(output)
            output = torch.stack(output_list).mean(0)
        else:
            output, y_true = get_pred(meta_module, dataloader, params=params, embedding=embedding)
        _, y_pred = output.topk(1)
        loss = F.cross_entropy(output, y_true)

    acc = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    meta_module.train()
    return acc, loss.item()

def log_metric(log_dict, prefix=""):
    prefixed_dict = dict()
    for key in log_dict:
        if not isinstance(log_dict[key], list):
            prefixed_dict[prefix+key] = log_dict[key]

    wandb.log(prefixed_dict)

    for key in log_dict:
        if isinstance(log_dict[key], list):
            for i in range(len(log_dict[key])):
                wandb.log({prefix+key: log_dict[key][i]})

def n_shot_trials_run(model_training, n_shot_trial_dict, config, log_name, data, *args, **kwargs):
    if "n_shot_trials_maxN" in config and config["n_shot_trials_maxN"] is not None:
        filtered_data, tasks_idx = filter_data_by_max_n_shot(data, config["n_shot_trials_maxN"])
        for n_shot in range(1, config["n_shot_trials_maxN"]+1):
            n_shot_trial_dict[f"{log_name}n_shot_{n_shot}/"] = {}
            log_dict = model_training.run_epoch(filtered_data, *args, n_shot_training=n_shot, tasks_idxs=tasks_idx, **kwargs)
            append_dict(n_shot_trial_dict[f"{log_name}n_shot_{n_shot}/"], log_dict)
    # full few shot
    log_dict = model_training.run_epoch(data, *args, n_shot_training="full", **kwargs)
    log_metric(log_dict, f"{log_name}few_shot/")

def filter_data_by_fraction(data, keep_tasks_frac):
    task_idx = np.range(len(list(data.keys()))*keep_tasks_frac)
    return data, task_idx

def filter_data_by_max_n_shot(data, max_k):
    data = copy.deepcopy(data)
    tasks_idx =[]
    for i,t in enumerate(list(data.keys())):
        ans, count = np.unique([a for [_, a] in data[t]["train"]], return_counts=True)
        filtered_ans = ans[count >= max_k]
        data[t]["train"] = [d for d in data[t]["train"] if d[1] in filtered_ans]
        data[t]["test"] = [d for d in data[t]["test"] if d[1] in filtered_ans]
        if len(ans)>=2:
            tasks_idx.append(i)
    return data, tasks_idx

def update_best_val(best_val_dict, meta_epoch, log_dict):
    if best_val_dict["best_val_accuracy"] < log_dict["query_accuracy_end"]:
        best_val_dict["best_val_accuracy"] = log_dict["query_accuracy_end"]
        best_val_dict["best_val_epoch"] = meta_epoch

def run_evals(model_training, train_data, test_data, n_shot_trial_dict, config, log_name, *args, skip_train=False, n_shot_from_opt_latent=False, best_val_dict=None, precomputed_latent_train=None, precomputed_latent_val=None, guided_inner=False, use_vae=False, **kwargs):
    # Eval on train data
    if not skip_train:
        log_dict = model_training.run_epoch(train_data, *args, guided_inner=guided_inner, keep_tasks_frac=config["diffusion_keep_tasks_frac"], precomputed_latent=precomputed_latent_train, use_vae=use_vae if precomputed_latent_train is None else False, **kwargs)
        log_metric(log_dict, log_name.format("train"))

    # Eval on test data
    log_dict, _, _, opt_latents = model_training.run_epoch(test_data, *args, guided_inner=guided_inner, precomputed_latent=precomputed_latent_val, use_vae=use_vae if precomputed_latent_val is None else False, output_mean_std=True, **kwargs)
    log_metric(log_dict, log_name.format("val"))
    if best_val_dict is not None:
        update_best_val(best_val_dict, kwargs["epoch"], log_dict)
        log_metric(best_val_dict)

    # N-shot eval on test data
    n_shot_trials_run(model_training, n_shot_trial_dict, config, log_name.format("val"), test_data, *args, opt_latents_for_n_shot=opt_latents if n_shot_from_opt_latent else None, use_vae=use_vae, **kwargs)