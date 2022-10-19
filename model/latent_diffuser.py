import torch
from torch import nn
from utils.diffusion_utils import timestep_embedding


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(nn.Linear(input_dim, output_dim),
                                    nn.LayerNorm(output_dim),
                                    nn.ReLU())
    
    def forward(self, x):
        return self.layers(x)


class LatentDiffuser(nn.Module):
    def __init__(self, x_dim=128, cond_emb_dim=768, timestep_emb_dim=100, hidden_dims=[128,128]):
        super().__init__()
        self.x_dim = x_dim
        self.cond_emb_dim = cond_emb_dim
        self.timestep_emb_dim = timestep_emb_dim
        self.hidden_dims = hidden_dims
        layers = [FeedForwardBlock(x_dim + cond_emb_dim + timestep_emb_dim, hidden_dims[0])]
        for i, h in enumerate(hidden_dims[1:]):
            layers += [FeedForwardBlock(hidden_dims[i], h)]
        layers += [nn.Linear(hidden_dims[-1], x_dim)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, t, cond_emb):
        if type(t) == int:
            t = torch.tensor(t, device = x.get_device()).repeat((x.shape[0], 1))
        # right now we only support basic concat of cond_emb and timestep_emb to input x
        timestep_emb = timestep_embedding(t.flatten(), self.timestep_emb_dim)
        eps = torch.cat((x, cond_emb, timestep_emb), 1)
        for layer in self.layers:
            eps = layer(eps)
        return eps

class SELayer(nn.Module):
    def __init__(self, input_dim, reduction=8):
        super().__init__()
        self.input_dim = input_dim
        self.reduction = reduction
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class LatentDiffuserV2(nn.Module):
    def __init__(self, x_dim=128, cond_emb_dim=768, timestep_emb_dim=100, hidden_dims=[128,128], se_reduction=8):
        super().__init__()
        self.x_dim = x_dim
        self.cond_emb_dim = cond_emb_dim
        self.timestep_emb_dim = timestep_emb_dim
        self.hidden_dims = hidden_dims
        self.se_reduction = se_reduction
        layers = [FeedForwardBlock(x_dim + cond_emb_dim + timestep_emb_dim, hidden_dims[0])]
        res_maps = [nn.Linear(x_dim, hidden_dims[0])]
        se_layers = [SELayer(x_dim)]
        for i, h in enumerate(hidden_dims[1:]):
            layers += [FeedForwardBlock(hidden_dims[i] + cond_emb_dim + timestep_emb_dim, h)]
            res_maps += [nn.Linear(hidden_dims[i], h)]
            se_layers += [SELayer(hidden_dims[i], reduction=self.se_reduction)]
        self.layers = nn.ModuleList(layers)
        self.res_maps = nn.ModuleList(res_maps)
        self.final_layer = nn.Linear(hidden_dims[-1], x_dim)

    def forward(self, x, t, cond_emb):
        if type(t) == int:
            t = torch.tensor(t, device = x.get_device()).repeat((x.shape[0], 1))
        # right now we only support basic concat of cond_emb and timestep_emb to input x
        timestep_emb = timestep_embedding(t.flatten(), self.timestep_emb_dim)
        eps_in = x
        for i in range(len(self.layers)):
            eps = self.layers[i](torch.cat((eps_in, cond_emb, timestep_emb), 1))
            eps_in = eps + self.res_maps[i](eps_in)
        eps = self.final_layer(eps)
        return eps
