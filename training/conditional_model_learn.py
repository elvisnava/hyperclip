import copy
from collections import OrderedDict

import numpy as np
import torch
import wandb
from data.dataloader.clip_vqa import CLIP_VQA
from data.dataloader.coco_tasks import COCO_Tasks
from model.custom_hnet import CLIPAdapter, EmbeddingModule, MetaModel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import clip_utils
from utils.misc_utils import append_dict, mean_dict
from utils.train_utils import get_pred, log_metric, test_accuracy

from training.vae_learn import sample_from_enc


class ConditionalModelTraining():
    # Train a "conditional" model from question embeddings and finetuned network features (weights, latents)
    def __init__(self, meta_module, cond_model, optimizer, image_features, text_features, ques_emb, config, device, net_feature_dim,
                hnet_gen, hnet_enc, train_on_vae=False, compute_hessian=False,
                 coco_categories=None, coco_answer_features=None, extend_coco_size=10 * 870):
        self.meta_module = meta_module
        self.cond_model = cond_model
        self.optimizer = optimizer
        self.image_features = image_features
        self.text_features = text_features
        self.ques_emb = ques_emb
        self.config = config
        self.device = device
        self.net_feature_dim = net_feature_dim
        self.compute_hessian = compute_hessian
        self.hnet_gen=hnet_gen
        self.hnet_enc=hnet_enc
        self.base_module=meta_module
        self.train_on_vae=train_on_vae

        self.coco_categories=coco_categories
        self.coco_answer_features=coco_answer_features
        self.extend_coco_size=extend_coco_size

    def reset(self, batch_size):
        self.sub_batch_i = 0
        self.ques_batch = torch.zeros(batch_size, clip_utils.embedding_size[self.config["clip_model"]], dtype=torch.float32).to(self.device)
        self.net_batch = torch.zeros(batch_size, self.net_feature_dim, dtype=torch.float32).to(self.device)
        self.task_batch = torch.zeros(batch_size, 1, dtype=torch.int).to(self.device)
        self.coco_batch = torch.zeros(batch_size, 1, dtype=torch.bool).to(self.device)

        self.hessian_batch = None
        if self.compute_hessian:
            self.hessian_batch = torch.zeros(batch_size, self.net_feature_dim, self.net_feature_dim, dtype=torch.float32).to(self.device)

    def reset_task(self, use_vae):
        if self.hnet_gen is not None and use_vae:
            embedding = sample_from_enc(self.hnet_enc, self.base_module.get_mainnet_weights(params = None).detach()).detach()

            enet = EmbeddingModule(self.hnet_gen.input_dim).to(self.device)

            enet.reset(embedding=embedding)
            mnet = CLIPAdapter(e_dim=self.meta_module.mnet.e_dim,
                                hidden_layers=[self.meta_module.mnet.hidden_size],
                                use_bias=self.meta_module.mnet.use_bias,
                                straight_through=self.meta_module.mnet.straight_through,
                                no_weights=True).to(self.device)

            self.meta_module = MetaModel(mnet=mnet, hnet=self.hnet_gen, enet=enet, config=self.config)
        else:
            self.meta_module = self.base_module


    def run_iter(self, task_idx, train_dataloader, net_features, train=True, clip_embedding=None, embed_hessian=None, coco=False, **kwargs):
        log_dict=dict()
        self.ques_batch[self.sub_batch_i] = clip_embedding if clip_embedding is not None else iter(train_dataloader).next()["ques_emb"][0]
        self.net_batch[self.sub_batch_i] = net_features
        self.task_batch[self.sub_batch_i] = task_idx
        self.coco_batch[self.sub_batch_i] = coco
        if self.compute_hessian:
            self.hessian_batch[self.sub_batch_i] = embed_hessian

        if self.sub_batch_i == self.ques_batch.shape[0]-1: # Train/test one batch of hyperclip once the batch is filled
            self.train_step(log_dict, **kwargs) if train else self.test_step(log_dict, **kwargs)

        self.sub_batch_i = (self.sub_batch_i+1) % self.ques_batch.shape[0]
        return log_dict

    def train_step(self, log_dict, **kwargs):
        raise NotImplementedError("Use a subclass of ConditionalModelTraining")

    def test_step(self, log_dict, **kwargs):
        raise NotImplementedError("Use a subclass of ConditionalModelTraining")

    def run_epoch(self, data, inner_epochs, inner_optim_fct, batch_size=32,
                  train_subtype="train", val_subtype="test", guided_inner=False,
                  precomputed_latent=None,
                  train=False, skip_cond=False, output_mean_std=False, tasks_idxs=None,
                  n_shot_training=None, opt_latents_for_n_shot=None,
                  debug=False, use_vae=False, keep_tasks_frac=1, extend_coco=False,
                  extend_coco_frac_train=0.5, num_ensemble=1,# frac of tasks to replace with extended coco
                  epoch=0, **kwargs):
        tasks = list(data.keys())
        if tasks_idxs is None:
            tasks_idxs = np.arange(len(tasks))

        if batch_size > len(tasks)*keep_tasks_frac:
            batch_size = int(len(tasks)*keep_tasks_frac)
            print("Warning: batch size too big, decreasing to {}".format(batch_size))

        if train:
            self.reset(batch_size)
        else:
            self.reset(len(data.keys()))

        if inner_epochs is not None:
            inner_epochs_range = [inner_epochs] if type(inner_epochs) == int else [int(i) for i in inner_epochs.split(",")]
        else:
            inner_epochs_range = None
        log_dict = dict()

        enable_coco = [False] * len(tasks)
        if precomputed_latent is None or "task_idx" not in precomputed_latent:
            shuffled_train_tasks = [tasks_idxs[idx] for idx in torch.randperm(len(tasks_idxs))]
            shuffled_coco_tasks = torch.randperm(self.extend_coco_size)
            shuffle_for_extended_coco_replace = torch.randperm(len(tasks))
            if extend_coco:
                enable_coco=[shuffle_for_extended_coco_replace[i] < extend_coco_frac_train * len(tasks) for i in range(len(tasks))]
        else:
            shuffled_train_tasks=precomputed_latent["task_idx"][0].long()
            shuffled_coco_tasks=precomputed_latent["task_idx"][0].long()
            if "coco" in precomputed_latent:
                enable_coco=precomputed_latent["coco"][0]

        if output_mean_std:
            all_tasks_optimized_params = torch.zeros((len(tasks), self.net_feature_dim)).to(self.device)
        n_shot_seed = np.random.randint(0, 1000000)
        n_corr_guesses_support_start = 0
        n_corr_guesses_support_end = 0
        n_corr_guesses_query_start = 0
        n_corr_guesses_query_end = 0
        n_tot_samples_support = 0
        n_tot_samples_query = 0

        for inner_train_iter in tqdm(range(len(shuffled_train_tasks))):
            self.reset_task(use_vae)
            curr_log_dict = dict()
            if enable_coco[inner_train_iter]:
                task_idx = shuffled_coco_tasks[inner_train_iter]
                train_dataset = COCO_Tasks(categories=self.coco_categories,
                                           dataSubType=train_subtype,
                                           image_features=self.image_features,
                                           coco_answer_features=self.coco_answer_features,
                                           task_seed=task_idx)
                test_dataset = COCO_Tasks(categories=self.coco_categories,
                                          dataSubType=val_subtype,
                                          image_features=self.image_features,
                                          coco_answer_features=self.coco_answer_features,
                                          task_seed=task_idx)
            else:
                task_idx = shuffled_train_tasks[inner_train_iter]
                if task_idx > len(tasks)*keep_tasks_frac:
                    continue
                train_dataset = CLIP_VQA(meta_data=data,
                                        dataSubType=train_subtype,
                                        task=tasks[task_idx],
                                        image_features=self.image_features,
                                        text_features=self.text_features,
                                        ques_emb=self.ques_emb,
                                        n_shot=n_shot_training)
                test_dataset = CLIP_VQA(meta_data=data,
                                        dataSubType=val_subtype,
                                        task=tasks[task_idx],
                                        image_features=self.image_features,
                                        text_features=self.text_features,
                                        ques_emb=self.ques_emb)

            train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

            # Inner loop
            if inner_epochs_range is not None:
                inner_epochs_sampled = inner_epochs_range[0] if len(inner_epochs_range) == 1 else \
                    np.random.randint(inner_epochs_range[0], inner_epochs_range[1]+1)
            else:
                inner_epochs_sampled = None

            embed_hessian=None

            init_inner_params = self.meta_module.get_inner_params()
            # Make sure to clone and detach to not optimize the actual initialization.
            inner_params = {k: v.clone().detach().requires_grad_() for (k,v) in self.meta_module.get_inner_params().items()}
            if opt_latents_for_n_shot is not None:
                assert self.meta_module.mnet.no_weight, "Not implemented  when meta_module is a mainnet."
                inner_params["enet.embedding"] = opt_latents_for_n_shot[task_idx].clone().detach().requires_grad_()

            train_start_acc, train_start_loss = test_accuracy(self.meta_module, train_dataloader, params=inner_params)
            val_start_acc, val_start_loss = test_accuracy(self.meta_module, test_dataloader, params=inner_params)
            
            inner_params_list=None
            if guided_inner:
                if num_ensemble>1:
                    inner_params_list=[]
                    for _ in range(num_ensemble):
                        inner_params = {k: v.clone().detach().requires_grad_() for (k,v) in self.meta_module.get_inner_params().items()}
                        inner_params_list.append(inner_params)
                        curr_log_dict.update(self.guided_inner(train_dataloader, inner_params, init_inner_params, inner_optim_fct,
                                  inner_train_iter, inner_epochs_sampled, batch_size, debug=inner_train_iter==1, **kwargs))
                else:
                    curr_log_dict.update(self.guided_inner(train_dataloader, inner_params, init_inner_params, inner_optim_fct,
                                         inner_train_iter, inner_epochs_sampled, batch_size, debug=inner_train_iter==1, **kwargs))
            else:
                if precomputed_latent is None or n_shot_training is not None:
                    inner_optimizer = inner_optim_fct(list(inner_params.values()))
                    for _ in range(inner_epochs_sampled):
                        outputs, labels = get_pred(self.meta_module, train_dataloader, params=inner_params)
                        inner_loss = F.cross_entropy(outputs, labels)
                        if debug and (self.sub_batch_i+1) % batch_size == 0:
                            wandb.log({"debug_inner_loss": inner_loss.item()})
                        inner_optimizer.zero_grad()
                        inner_loss.backward()
                        inner_optimizer.step()

                    if self.compute_hessian:
                        embed_hessian = self.compute_feature_hessian(train_dataloader, self.meta_module, inner_params["enet.embedding"]).squeeze()

                else:
                    # If i use the precomputed latents, I "simulate" the finetuning inner loop after it's supposed
                    # to happen, to not break metrics and mean/std calculations (super ugly, to refactor)
                    if self.meta_module.mnet.no_weight:
                        inner_params["enet.embedding"] = precomputed_latent["embedding"][0, inner_train_iter]
                    else:
                        inner_params.update({"mnet."+k:v for (k,v) in self.meta_module.mnet.load_from_vector(precomputed_latent["w_vect"][0, inner_train_iter]).items()})

                    if self.compute_hessian:
                        embed_hessian = precomputed_latent["hessian"][0, inner_train_iter]

            # Train set accuracy
            if inner_params_list is not None:
                train_end_acc, train_end_loss = test_accuracy(self.meta_module, train_dataloader, params_list=inner_params_list)
                val_end_acc, val_end_loss = test_accuracy(self.meta_module, test_dataloader, params_list=inner_params_list)
                
                avr_inner_params = OrderedDict()
                avr_inner_params.update({k: torch.stack([p[k] for p in inner_params_list]).mean(0).detach() for k in inner_params.keys()})
            
                train_end_acc_avr, train_end_loss_avr = test_accuracy(self.meta_module, train_dataloader, params=avr_inner_params)
                val_end_acc_avr, val_end_loss_avr = test_accuracy(self.meta_module, test_dataloader, params=avr_inner_params)

                curr_log_dict["support_loss_end_avr"] = train_end_loss_avr
                curr_log_dict["query_loss_end_avr"] = val_end_loss_avr

                curr_log_dict["support_accuracy_end_avr"] = train_end_acc_avr
                curr_log_dict["query_accuracy_end_avr"] = val_end_acc_avr

            else:
                train_end_acc, train_end_loss = test_accuracy(self.meta_module, train_dataloader, params=inner_params)
                val_end_acc, val_end_loss = test_accuracy(self.meta_module, test_dataloader, params=inner_params)

            n_tot_samples_support += len(train_dataset)
            n_tot_samples_query += len(test_dataset)
            curr_log_dict["query_accuracy_start"] = val_start_acc
            n_corr_guesses_query_start += val_start_acc * len(test_dataset)
            curr_log_dict["query_accuracy_end"] = val_end_acc
            n_corr_guesses_query_end += val_end_acc * len(test_dataset)
            curr_log_dict["support_accuracy_start"] = train_start_acc
            n_corr_guesses_support_start += train_start_acc * len(train_dataset)
            curr_log_dict["support_accuracy_end"] = train_end_acc
            n_corr_guesses_support_end += train_end_acc * len(train_dataset)
            curr_log_dict["query_loss_start"] = val_start_loss
            curr_log_dict["query_loss_end"] = val_end_loss
            curr_log_dict["support_loss_start"] = train_start_loss
            curr_log_dict["support_loss_end"] = train_end_loss

            # Actually run cond model training/eval
            if not skip_cond:
                cond_dict = self.run_iter(task_idx, train_dataloader,
                                          self.feed_net_feature(inner_params = inner_params),embed_hessian=embed_hessian,
                                          train=train, coco=enable_coco[inner_train_iter],
                                          clip_embedding=precomputed_latent["clip_embedding"][0, inner_train_iter], **kwargs)
                curr_log_dict.update(cond_dict)

            if output_mean_std:
                all_tasks_optimized_params[task_idx] = self.feed_net_feature(inner_params = inner_params)

            append_dict(log_dict, curr_log_dict)

            if debug and self.sub_batch_i % batch_size == 0:
                output_dict = mean_dict(log_dict)
                output_dict["query_accuracy_start_flatten"] = n_corr_guesses_query_start / n_tot_samples_query
                output_dict["query_accuracy_end_flatten"] = n_corr_guesses_query_end / n_tot_samples_query
                output_dict["support_accuracy_start_flatten"] = n_corr_guesses_support_start / n_tot_samples_support
                output_dict["support_accuracy_end_flatten"] = n_corr_guesses_support_end / n_tot_samples_support
                log_metric(output_dict, prefix="debug_")
                log_dict = dict()

        output_dict = mean_dict(log_dict)
        output_dict["query_accuracy_start_flatten"] = n_corr_guesses_query_start / n_tot_samples_query
        output_dict["query_accuracy_end_flatten"] = n_corr_guesses_query_end / n_tot_samples_query
        output_dict["support_accuracy_start_flatten"] = n_corr_guesses_support_start / n_tot_samples_support
        output_dict["support_accuracy_end_flatten"] = n_corr_guesses_support_end / n_tot_samples_support
        output_dict["epoch"] = epoch

        if not output_mean_std:
            return output_dict
        else:
            optimized_params_mean = torch.mean(all_tasks_optimized_params, dim=0, keepdim=True)
            optimized_params_std = torch.std(all_tasks_optimized_params, dim=0, keepdim=True)
            return output_dict, optimized_params_mean, optimized_params_std, all_tasks_optimized_params

    def guided_inner(self, train_dataloader, inner_params, init_inner_params, inner_optim_fct,
                     inner_train_iter, inner_epochs, batch_size, debug, **kwargs):
        raise NotImplementedError("Use a subclass of ConditionalModelTraining")

    def feed_net_feature(self, **kwargs):
        raise NotImplementedError("Use a subclass of ConditionalModelTraining")

    def compute_feature_hessian(self, train_dataloader, meta_module, embedding):
        """ Only supports when the embedding is the net feature. Would generalize this if needed. """
        def get_nll(embedding):
            curr_params = OrderedDict()
            curr_params["enet.embedding"] = embedding
            outputs, labels = get_pred(meta_module, train_dataloader, params=curr_params)
            inner_loss = F.cross_entropy(outputs, labels)
            return inner_loss

        return torch.autograd.functional.hessian(get_nll, embedding)

