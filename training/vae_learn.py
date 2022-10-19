import math
from collections import OrderedDict

import torch
import wandb
from data.dataloader.clip_vqa import CLIP_VQA
from model.custom_hnet import CLIPAdapter, EmbeddingModule, MetaModel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc_utils import append_dict, mean_dict
from utils.train_utils import get_pred, log_metric, test_accuracy


def sample_from_enc(encoder, real_imgs):
    vect = encoder(real_imgs)
    enc_dim = vect.shape[1] // 2
    mu, logvar = vect[:, :enc_dim], vect[:, enc_dim:] - 1
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    encoding = eps * std + mu
    return encoding


class VAETraining():
    def __init__(self, meta_module, hnet_gen, hnet_enc, optimizer_gen, optimizer_enc, image_features, text_features,
                 ques_emb, config, device):
        self.meta_module = meta_module
        self.hnet_gen = hnet_gen
        self.hnet_enc = hnet_enc
        self.optimizer_gen = optimizer_gen
        self.optimizer_enc = optimizer_enc

        self.image_features = image_features
        self.text_features = text_features
        self.ques_emb = ques_emb
        self.config = config
        self.device=device
        self.feature_mean=None
        self.feature_std=None

    def set_stats(self, mean, std):
        self.feature_mean= mean
        self.feature_std=std

    def reset(self, batch_size, device):
        self.vae_sub_batch_i = 0
        self.weight_batch = torch.zeros(batch_size, self.hnet_enc.input_dim, dtype=torch.float32).to(device)

    def train(self, model_weights):
        log_dict=dict()

        self.weight_batch[self.vae_sub_batch_i] = model_weights

        if self.vae_sub_batch_i == self.weight_batch.shape[0]-1:
            real_imgs = self.weight_batch
            encoder=self.hnet_enc
            generator=self.hnet_gen
            kld_weight=self.config["kld_weight"]

            vect = encoder(real_imgs)
            enc_dim = vect.shape[1] // 2
            mu, logvar = vect[:, :enc_dim], vect[:, enc_dim:] - 1
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            factor = math.sqrt(real_imgs.shape[-1])

            encoding = eps * std + mu
            generated_imgs = generator(encoding)

            recons = (generated_imgs - real_imgs)

            if self.feature_std is not None:
                recons = recons / self.feature_std

            recons_loss = recons.pow(2).mean(0).sum() / factor

            kld_loss = kld_weight * torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1),
                                               dim=0) / factor

            loss = (recons_loss + kld_loss)


            self.optimizer_gen.zero_grad()
            self.optimizer_enc.zero_grad()
            loss.backward()
            log_dict = {"loss": loss.item(), "kld_loss": kld_loss.item(), "recons_loss": recons_loss.item(),
                        "encoding_norm": encoding.pow(2).mean().sqrt().item(),
                        "norm_fake": generated_imgs.norm(dim=1).mean(dim=0).item(),
                        "norm_real": real_imgs.norm(dim=1).mean(dim=0).item(), 
                        "grad_norm_enc": torch.stack([p.grad.pow(2).sum() for p in encoder.parameters() if p.grad is not None]).sum().sqrt().item(),
                        "grad_norm_gen": torch.stack([p.grad.pow(2).sum() for p in generator.parameters() if p.grad is not None]).sum().sqrt().item()}


            if self.config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=self.config["grad_clip"])
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=self.config["grad_clip"])

            log_dict.update({"grad_norm_enc_clipped": torch.stack([p.grad.pow(2).sum() for p in encoder.parameters() if p.grad is not None]).sum().sqrt().item(),
                        "grad_norm_gen_clipped": torch.stack([p.grad.pow(2).sum() for p in generator.parameters() if p.grad is not None]).sum().sqrt().item()})

            self.optimizer_gen.step()
            self.optimizer_enc.step()

        self.vae_sub_batch_i = (self.vae_sub_batch_i + 1) % self.weight_batch.shape[0]
        return log_dict

    def run_epoch(self, data, inner_epochs, inner_optim_fct, batch_size=32, device=None,
                  train_subtype="train", val_subtype="test",
                  train=False, generated=False, reconstructed=False, debug=False, precomputed_latent=None, output_mean_std=False, epoch=0):
        if train:
            self.reset(batch_size, device)

        log_dict = dict()
        tasks = list(data.keys())
        if precomputed_latent is None or "task_idx" not in precomputed_latent:
            shuffled_train_tasks = torch.randperm(len(tasks))
        else:
            shuffled_train_tasks=precomputed_latent["task_idx"][0]

        if output_mean_std:
            all_tasks_optimized_params = torch.zeros((len(tasks), self.hnet_enc.input_dim)).to(self.device)

        for inner_train_iter in tqdm(range(len(tasks))):
            curr_log_dict = dict()
            task_idx = shuffled_train_tasks[inner_train_iter]
            train_dataset = CLIP_VQA(meta_data=data,
                                     dataSubType=train_subtype,
                                     task=tasks[task_idx],
                                     image_features=self.image_features,
                                     text_features=self.text_features,
                                     ques_emb=self.ques_emb)
            test_dataset = CLIP_VQA(meta_data=data,
                                    dataSubType=val_subtype,
                                    task=tasks[task_idx],
                                    image_features=self.image_features,
                                    text_features=self.text_features,
                                    ques_emb=self.ques_emb)

            train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

            # task_embedding = iter(train_dataloader).next()["ques_emb"][0]
            if generated or reconstructed:
                embedding = sample_from_enc(self.hnet_enc, self.meta_module.get_mainnet_weights().detach()).detach()
                enet = EmbeddingModule(self.hnet_gen.input_dim).to(device)
                enet.reset(embedding=embedding)

                mnet = CLIPAdapter(e_dim=self.meta_module.mnet.e_dim,
                                    hidden_layers=[self.meta_module.mnet.hidden_size],
                                    use_bias=self.meta_module.mnet.use_bias,
                                    straight_through=self.meta_module.mnet.straight_through,
                                    no_weights=True).to(device)
                meta_module=MetaModel(mnet=mnet, hnet=self.hnet_gen, enet=enet, config=self.config)
            elif False: #reconstructed:
                mnet = CLIPAdapter(e_dim=self.meta_module.mnet.e_dim,
                                    hidden_layers=[self.meta_module.mnet.hidden_size],
                                    use_bias=self.meta_module.mnet.use_bias,
                                    straight_through=self.meta_module.mnet.straight_through,
                                    no_weights=True).to(device)
                meta_module=MetaModel(mnet=mnet, hnet=self.hnet_gen, enet=None, config=self.config)
            else:
                meta_module = self.meta_module

            # Make sure to clone and detach to not optimize the actual initialization.
            inner_params = {k: v.clone().detach().requires_grad_() for (k,v) in meta_module.get_inner_params().items()}

            if reconstructed:
                tmp_inner_params = OrderedDict()
                tmp_inner_params["enet.embedding"] = sample_from_enc(self.hnet_enc, self.meta_module.get_mainnet_weights()).detach()
                inner_params=tmp_inner_params
                #task_embedding = sample_from_enc(self.hnet_enc, self.meta_module.get_mainnet_weights(params=inner_params)).detach()

            train_start_acc, train_start_loss = test_accuracy(meta_module, train_dataloader,
                                                                   params=inner_params) #, embedding = task_embedding)
            val_start_acc, val_start_loss = test_accuracy(meta_module, test_dataloader, params=inner_params) #, embedding = task_embedding)

            if precomputed_latent is None or generated:
                # Inner loop
                inner_optimizer = inner_optim_fct(list(inner_params.values()))
                for _ in range(inner_epochs):
                    outputs, labels = get_pred(meta_module, train_dataloader, params=inner_params)
                    inner_loss = F.cross_entropy(outputs, labels)
                    if debug and inner_train_iter % batch_size == 0:
                        wandb.log({"debug_inner_loss": inner_loss.item()})
                    inner_optimizer.zero_grad()
                    inner_loss.backward()
                    inner_optimizer.step()
            else:
                if self.meta_module.mnet.no_weight:
                    inner_params["enet.embedding"] = precomputed_latent["embedding"][0, inner_train_iter]
                else:
                    inner_params.update({"mnet."+k:v for (k,v) in self.meta_module.mnet.load_from_vector(precomputed_latent["w_vect"][0, inner_train_iter]).items()})

            if reconstructed:
                tmp_inner_params = OrderedDict()
                tmp_inner_params["enet.embedding"] = sample_from_enc(self.hnet_enc, self.meta_module.get_mainnet_weights(params=inner_params)).detach()
                inner_params=tmp_inner_params
                #task_embedding = sample_from_enc(self.hnet_enc, self.meta_module.get_mainnet_weights(params=inner_params)).detach()

            # Train set accuracy
            train_end_acc, train_end_loss = test_accuracy(meta_module, train_dataloader, params=inner_params) #, embedding = task_embedding)
            val_end_acc, val_end_loss = test_accuracy(meta_module, test_dataloader, params=inner_params) #, embedding = task_embedding)

            curr_log_dict["query_accuracy_start"] = val_start_acc
            curr_log_dict["query_accuracy_end"] = val_end_acc
            curr_log_dict["support_accuracy_start"] = train_start_acc
            curr_log_dict["support_accuracy_end"] = train_end_acc
            curr_log_dict["query_loss_start"] = val_start_loss
            curr_log_dict["query_loss_end"] = val_end_loss
            curr_log_dict["support_loss_start"] = train_start_loss
            curr_log_dict["support_loss_end"] = train_end_loss

            if train:
                vae_dict = self.train(self.meta_module.get_mainnet_weights(params=inner_params).detach())
                curr_log_dict.update(vae_dict)

            append_dict(log_dict, curr_log_dict)

            if debug and inner_train_iter % batch_size == 0:
                log_metric(mean_dict(log_dict), prefix="debug_")
                log_dict = dict()

            if output_mean_std:
                all_tasks_optimized_params[inner_train_iter] = self.meta_module.get_mainnet_weights(params=inner_params).detach()

        output_dict = mean_dict(log_dict)
        output_dict["epoch"] = epoch

        if not output_mean_std:
            return output_dict
        else:
            optimized_params_mean = torch.mean(all_tasks_optimized_params, dim=0, keepdim=True)
            optimized_params_std = torch.std(all_tasks_optimized_params, dim=0, keepdim=True)
            return output_dict, optimized_params_mean, optimized_params_std, all_tasks_optimized_params

