import math

import torch
import wandb
from torch.nn import functional as F
from utils.train_utils import get_pred, test_accuracy

from training.conditional_model_learn import ConditionalModelTraining
from training.vae_learn import sample_from_enc


class HyperclipTraining(ConditionalModelTraining):
    def __init__(self, meta_module, hnet_gen, hnet_enc, hyperclip, optimizer,
                 image_features, text_features, ques_emb, config, device, train_on_vae=False, **kwargs):
        net_feature_dim = hyperclip.hyper.input_dim
        super().__init__(meta_module, hyperclip, optimizer, image_features, text_features, ques_emb,
                         config, device, net_feature_dim, hnet_gen, hnet_enc, train_on_vae=train_on_vae, **kwargs)
        self.feature_mean=None
        self.feature_std=None

    def train_step(self, log_dict):
        self.optimizer.zero_grad()

        net_batch = self.net_batch
        if self.feature_mean is not None:
            net_batch = net_batch - self.feature_mean
            net_batch = net_batch / self.feature_std

        _, hyperclip_loss = self.cond_model.forward(hyper = net_batch, text = self.ques_batch, precomputed_it_embs = True)
        log_dict.update({"hyperclip_loss": hyperclip_loss.detach().cpu().numpy()})
        hyperclip_loss.backward()
        self.optimizer.step()

    def test_step(self, log_dict):
        with torch.no_grad():
            self.cond_model.eval()

            net_batch = self.net_batch
            if self.feature_mean is not None:
                net_batch = net_batch-self.feature_mean
                net_batch = net_batch / self.feature_std

            _, hyperclip_loss = self.cond_model.forward(hyper = net_batch,
                                                        text = self.ques_batch, precomputed_it_embs = True)
            log_dict.update({"hyperclip_val_loss": hyperclip_loss.detach().cpu().numpy()})
            self.cond_model.train()


    def set_stats(self, mean, std):
        self.feature_mean= mean
        self.feature_std=std


    def get_compute_grad(self, train_dataloader):
        def compute_grad(inner_params):
            with torch.enable_grad():
                x = inner_params["enet.embedding"]
                outputs, labels = get_pred(self.meta_module, train_dataloader, params=inner_params)
                inner_loss = F.cross_entropy(outputs, labels)
                grad =torch.autograd.grad(inner_loss, x)
                return grad[0]
        return compute_grad


    def guided_inner(self, train_dataloader, inner_params, init_inner_params, inner_optim_fct,
                     inner_train_iter, inner_epochs, batch_size, debug, langevin_eps=0, guidance_scheduler_fn=None, guidance_init_l2_weight=0,  **kwargs):

#        if eval:
        compute_grad_fn = self.get_compute_grad(train_dataloader)

        inner_optimizer = inner_optim_fct(list(inner_params.values()))
        if guidance_scheduler_fn is not None:
            guidance_scheduler = guidance_scheduler_fn(inner_optimizer)
        else:
            guidance_scheduler = None


        log_dict=dict()
        log_dict["cos_sim"]=[]
        log_dict["acc"]=[]
        log_dict["loss"]=[]

        for _ in range(inner_epochs):
            task_ques_emb = next(iter(train_dataloader))["ques_emb"][0]
            weights = self.meta_module.get_mainnet_weights(ques_emb=task_ques_emb, params=inner_params)

            if self.feature_mean is not None:
                weights = weights - self.feature_mean
                weights = weights / self.feature_std

            hyperclip = self.cond_model

            task_weight_emb = hyperclip.encode_hyper(weights)

            norm_task_ques_emb = task_ques_emb / task_ques_emb.norm(dim=-1, keepdim=True)
            norm_task_weight_emb = task_weight_emb / task_weight_emb.norm(dim=-1, keepdim=True)

            inner_product_embs_loss = - norm_task_weight_emb @ norm_task_ques_emb.T

            init_l2_loss = torch.stack(
                [(ip - p).pow(2).sum() for (ip, p)
                    in zip(init_inner_params.values(), inner_params.values())]).sum() / 2
            inner_loss = inner_product_embs_loss + init_l2_loss* guidance_init_l2_weight

            inner_optimizer.zero_grad()
            inner_loss.backward()

            if True: #eval:
                alignment = torch.nn.CosineSimilarity(dim=0)(inner_params["enet.embedding"].grad.view(-1), compute_grad_fn(inner_params).view(-1))
                log_dict["cos_sim"].append(alignment.item())
                a, l = test_accuracy(self.meta_module, train_dataloader, params=inner_params)
                log_dict["acc"].append(a)
                log_dict["loss"].append(l)

            if debug:
                wandb.log({"debug_inner_guidance_loss": inner_product_embs_loss.item(),
                            "debug_inner_l2_loss": init_l2_loss.item(),
                            "inner_lr": inner_optimizer.param_groups[0]['lr'], 
                            "gradnorm": torch.stack([p.grad.pow(2).sum() for p in inner_params.values() if p.grad is not None]).sum().item()})


            if langevin_eps>0:
                for p in inner_params.values():
                    p.grad.data += langevin_eps*torch.randn_like(p.grad.data)/math.sqrt(inner_optimizer.param_groups[0]['lr'])

            inner_optimizer.step()
            if guidance_scheduler is not None:
                guidance_scheduler.step()

        return log_dict


    def feed_net_feature(self, inner_params=None):
        if self.train_on_vae and self.hnet_gen is not None:
             return self.hnet_gen(sample_from_enc(self.hnet_enc, self.base_module.get_mainnet_weights(params = inner_params).detach())).detach()
        return self.base_module.get_mainnet_weights(params = inner_params).detach()
