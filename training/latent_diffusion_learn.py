from collections import OrderedDict

import torch
import wandb
from torch.nn import functional as F
from utils.train_utils import get_pred

from training.conditional_model_learn import ConditionalModelTraining
from training.vae_learn import sample_from_enc


class LatentDiffusionTraining(ConditionalModelTraining):
    def __init__(self,
                 meta_module,
                 diffusion_model,
                 optimizer,
                 n_timestep,
                 beta,
                 ema,
                 net_feature_dim,
                 train_data_for_mean_std,
                 inner_opt_for_mean_std,
                 image_features,
                 text_features,
                 ques_emb,
                 config,
                 device,
                 hnet_gen, hnet_enc, train_on_vae=False,
                 compute_hessian=False,
                 compute_latents_mean_std = True,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 class_free_guidance=True,
                 class_free_training_uncond_rate=0.2,
                 precomputed_latent_for_mean_std=None,
                 **kwargs
                 ):

        super().__init__(meta_module, diffusion_model, optimizer, image_features, text_features,
                         ques_emb, config, device, net_feature_dim, hnet_gen, hnet_enc, train_on_vae, compute_hessian, **kwargs)
        self.n_timestep = n_timestep
        self.beta = torch.tensor(beta).to(self.device)
        self.ema = ema
        
        self.alpha = (1 - self.beta)
        self.alpha_cumprod = self.alpha.log().cumsum(0).exp().float()
        self.alpha_cumprod_prev = torch.cat((torch.ones(1, device = self.device), self.alpha_cumprod[:-1]))
        self.v_posterior = v_posterior
        self.sigma = ((1 - self.v_posterior) * (self.beta * (1 - self.alpha_cumprod_prev) / (1 - self.alpha_cumprod))  + self.v_posterior * self.beta).sqrt().float()
        self.beta, self.alpha = self.beta.float(), self.alpha.float()

        self.class_free_guidance = class_free_guidance
        self.class_free_training_uncond_rate = class_free_training_uncond_rate

        if compute_latents_mean_std:
            _, self.latents_mean, self.latents_std, _ = self.run_epoch(train_data_for_mean_std, config["inner_epochs"], inner_opt_for_mean_std,
                                                                    batch_size=config["diffusion_batch_size"],
                                                                    train=True, train_subtype = config["train_subtype"], val_subtype=config["val_subtype"],
                                                                    precomputed_latent=precomputed_latent_for_mean_std,
                                                                    skip_cond=True, output_mean_std=True, keep_tasks_frac=config["diffusion_keep_tasks_frac"])
        else:
            self.latents_mean, self.latents_std = torch.zeros((1,self.net_feature_dim)).to(self.device), torch.ones((1,self.net_feature_dim)).to(self.device)

        wandb.log({"latents_mean": self.latents_mean.detach().cpu().numpy()})
        wandb.log({"latents_std": self.latents_std.detach().cpu().numpy()})

    def forward_diffusion(self, latents_batch, ques_batch, metric=None):
        norm_latents_batch = (latents_batch - self.latents_mean) / self.latents_std
        t = torch.randint(self.n_timestep, (norm_latents_batch.size(0),) + (1,) * (norm_latents_batch.dim() - 1), device = self.device)
        eps = torch.randn_like(norm_latents_batch)
        norm_latents_batch_t = torch.sqrt(self.alpha_cumprod[t]) * norm_latents_batch + torch.sqrt(1 - self.alpha_cumprod[t]) * eps
        output = self.cond_model(norm_latents_batch_t, t, ques_batch)
        if metric is not None:
            loss = (((eps - output).unsqueeze(1) @ metric @ (eps - output).unsqueeze(-1)) / metric.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)).mean()
        else:
            loss = (eps - output).pow(2).mean()
        return loss

    def train_step(self, log_dict):
        self.optimizer.zero_grad()

        ques_batch = torch.clone(self.ques_batch)
        if self.class_free_guidance:
            perm = torch.randperm(ques_batch.shape[0])
            idx = perm[:int(perm.shape[0]*self.class_free_training_uncond_rate)]
            ques_batch[idx] = torch.zeros(ques_batch.shape[1]).to(self.device)

        loss = self.forward_diffusion(self.net_batch, ques_batch, self.hessian_batch)

        log_dict.update({"diffusion_loss": loss.detach().cpu().numpy()})
        loss.backward()
        self.optimizer.step()

        if self.ema is not None: self.ema.step()

    def test_step(self, log_dict):
        with torch.no_grad():
            self.cond_model.eval()

            ques_batch = torch.clone(self.ques_batch)
            if self.class_free_guidance:
                perm = torch.randperm(ques_batch.shape[0])
                idx = perm[:int(perm.shape[0]*self.class_free_training_uncond_rate)]
                ques_batch[idx] = torch.zeros(ques_batch.shape[1]).to(self.device)

            loss = self.forward_diffusion(self.net_batch, ques_batch, self.hessian_batch)
            
            log_dict.update({"diffusion_val_loss": loss.detach().cpu().numpy()})
            self.cond_model.train()

    def get_compute_grad(self, train_dataloader):
        def compute_grad(x):
            with torch.enable_grad():
                inner_params = OrderedDict()
                x = x.detach().requires_grad_()
                inner_params["enet.embedding"] = x * self.latents_std + self.latents_mean
                outputs, labels = get_pred(self.meta_module, train_dataloader, params=inner_params)
                inner_loss = F.cross_entropy(outputs, labels)
                grad =torch.autograd.grad(inner_loss, x)
                return grad[0]
        return compute_grad

    def guided_inner(self, train_dataloader, inner_params, init_inner_params, inner_optim_fct,
                     inner_train_iter, inner_epochs, batch_size, debug, class_guidance_gamma=None,
                     init_guidance_at="random", guidance_start_from_t_frac=1, fast_sampling_factor=None, few_shot_guidance=False, few_shot_gamma=1, **kwargs):
        #IMPORTANT: x_accuracy_start printed in wandb is not the one for the random initialized embedding (to which the diffusion is applied to)
        # but the one for the initial emb initialization we are NOT using (except if we start from pre-trained !)
        if few_shot_guidance:
            compute_grad_fn = self.get_compute_grad(train_dataloader)
        else:
            compute_grad_fn=None
        if class_guidance_gamma is None:
            class_guidance_gamma = 1.
        if init_guidance_at=="random":
            latent_start_point = None
        elif init_guidance_at=="pre-trained":
            latent_start_point = init_inner_params["enet.embedding"].clone().detach()
        sampled_emb = self.generate(next(iter(train_dataloader))["ques_emb"][0],
                                    self.class_free_guidance,
                                    class_guidance_gamma,
                                    latent_start_point=latent_start_point,
                                    start_from_t_frac=guidance_start_from_t_frac,
                                    fast_sampling_factor=fast_sampling_factor, 
                                    compute_grad_fn=compute_grad_fn,
                                    few_shot_gamma=few_shot_gamma)
        inner_params["enet.embedding"] = sampled_emb  # refactor to be more general
        return dict()


    def generate(self, cond_emb, class_free_guidance, gamma, latent_start_point=None, start_from_t_frac=1., fast_sampling_factor=None, compute_grad_fn=None, few_shot_gamma=1):

        with torch.no_grad():

            alpha_cumprod = self.alpha_cumprod
            sigma = self.sigma
            if fast_sampling_factor is not None and fast_sampling_factor > 1:
                alpha_cumprod = self.alpha_cumprod[::fast_sampling_factor]
                alpha_cumprod_prev = torch.cat((torch.ones(1, device = self.device), alpha_cumprod[:-1]))
                beta = 1 - (alpha_cumprod / alpha_cumprod_prev)
                sigma = ((1 - self.v_posterior) * (beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))  + self.v_posterior * beta).sqrt().float()

            if latent_start_point is None:
                x = torch.randn(self.latents_mean.shape, device = self.device)
            else:
                x = (latent_start_point - self.latents_mean) / self.latents_std

            for t in range(int(sigma.shape[0]*start_from_t_frac)-1, -1, -1):
                t_for_model = t
                if fast_sampling_factor is not None and fast_sampling_factor > 1:
                    t_for_model = t * fast_sampling_factor
                if not class_free_guidance or gamma==1.:
                    output = self.cond_model(x, t_for_model, cond_emb)
                else:
                    output = (1 - gamma) * self.cond_model(x, t_for_model, torch.zeros_like(cond_emb)) + gamma * self.cond_model(x, t_for_model, cond_emb)

                if compute_grad_fn is not None:
                    grad = compute_grad_fn(x)
                    grad = grad / grad.norm()*output.norm()*few_shot_gamma
                    output = output + grad

                z = torch.zeros_like(x) if t == 0 else torch.randn_like(x)
                x = 1/torch.sqrt(self.alpha[t]) * (x - (1-self.alpha[t]) / torch.sqrt(1-alpha_cumprod[t]) * output) + sigma[t] * z

            x = x * self.latents_std + self.latents_mean

            return x

    def feed_net_feature(self, inner_params=None):
        if self.train_on_vae and self.hnet_gen is not None:
             return sample_from_enc(self.hnet_enc, self.meta_module.get_mainnet_weights(params = inner_params).detach()).detach()
        # works if inner_params contains only one element, the hnet/vae embedding
        return inner_params["enet.embedding"].detach()

