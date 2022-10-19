import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmeta.modules import MetaLinear, MetaModule, MetaSequential
from utils import clip_utils


def get_from_dict_or_default(module, dict, key):
    if key in dict:
        return dict[key]
    return getattr(module, key)

class Mlp(MetaModule):
    def __init__(self, input_dim, hidden_dims, output_dim, nonlin='relu'):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dims=hidden_dims
        self.output_dim=output_dim

        linear = []
        prev_layer_dim = self.input_dim
        for dim in hidden_dims:
            linear.append(MetaLinear(prev_layer_dim, dim))
            torch.nn.init.kaiming_normal_(linear[-1].weight, nonlinearity="relu")
            linear[-1].bias.data *= 0
            if nonlin == 'relu':
                linear.append(nn.ReLU())
            elif nonlin == 'tanh':
                linear.append(nn.Tanh())
            elif nonlin == 'softplus':
                linear.append(nn.Softplus())
            else:
                assert False

            prev_layer_dim = dim

        linear.append(MetaLinear(prev_layer_dim, output_dim))
        torch.nn.init.kaiming_normal_(linear[-1].weight, nonlinearity="linear")
        linear[-1].bias.data *= 0
        self.mlp = MetaSequential(*linear)

    def forward(self, uncond_input, params = None):
        if len(uncond_input.shape) == 1:
            uncond_input = uncond_input.unsqueeze(0)
        return self.mlp(uncond_input, params = self.get_subdict(params, "mlp"))

class HyperGenerator(Mlp):
    def __init__(self, mnet, e_dim, hidden_dims, normalize=False):
        super().__init__(e_dim, hidden_dims, mnet.get_parameter_vector().shape[0])
        self.normalize=normalize
        if self.normalize:
            self.register_buffer("feature_mean", torch.zeros(self.output_dim))
            self.register_buffer("feature_std", torch.ones(self.output_dim))

    def set_stats(self, mean, std):
        self.feature_mean.data = mean.detach()
        self.feature_std.data = std.detach()

    def forward(self, uncond_input, params = None):
        res = super().forward(uncond_input, params)
        if self.normalize:
            res = res * self.feature_std
            res = res + self.feature_mean
        return res


class HyperEncoder(Mlp):
    def __init__(self, mnet, e_dim, hidden_dims, normalize=False):
        super().__init__(mnet.get_parameter_vector().shape[0], hidden_dims, 2*e_dim)
        self.normalize=normalize
        if self.normalize:
            self.register_buffer("feature_mean", torch.zeros(self.input_dim))
            self.register_buffer("feature_std", torch.ones(self.input_dim))

    def set_stats(self, mean, std):
        self.feature_mean.data = mean.detach()
        self.feature_std.data = std.detach()

    def forward(self, uncond_input, params = None):
        if self.normalize:
            uncond_input = uncond_input - self.feature_mean
            uncond_input = uncond_input / self.feature_std
        return super().forward(uncond_input, params)


class HyperDiscriminator(Mlp):
    def __init__(self, mnet, hidden_dims):
        super().__init__(mnet.get_parameter_vector().shape[0], hidden_dims, 1)

class CLIPAdapter(MetaModule):
    def __init__(self, e_dim, hidden_layers, use_bias, no_weights=False, straight_through=False, ignore_passed_weights=False):
        super().__init__()
        assert len(hidden_layers) == 1, "Architecture supports a single hidden layer."
        hidden_size = hidden_layers[0]
        self.e_dim=e_dim
        self.hidden_size=hidden_size
        self.use_bias=use_bias
        self.no_weights=no_weights
        self.no_weight=no_weights
        self.straight_through=straight_through
        self.ignore_passed_weights = ignore_passed_weights

        if no_weights:
            self.register_buffer("W1", torch.randn(hidden_size, e_dim).requires_grad_())
            self.register_buffer("b1", torch.randn(hidden_size).requires_grad_() if use_bias else None)
            self.register_buffer("W2", torch.randn(e_dim, hidden_size).requires_grad_())
            self.register_buffer("b2", torch.randn(e_dim).requires_grad_() if use_bias else None)
        else:
            norm_W1=math.sqrt(self.e_dim) if self.straight_through else 1
            norm_W2=math.sqrt(self.hidden_size) if self.straight_through else 1

            self.W1=torch.nn.Parameter((torch.randn(hidden_size, e_dim)/norm_W1).requires_grad_())
            self.b1=torch.nn.Parameter(torch.randn(hidden_size).requires_grad_()) if use_bias else None
            self.W2=torch.nn.Parameter((torch.randn(e_dim, hidden_size)/norm_W2).requires_grad_())
            self.b2=torch.nn.Parameter(torch.randn(e_dim).requires_grad_()) if use_bias else None

    def get_parameter_vector(self, params=None):
        if params is not None:
            return torch.cat([params["W1"].flatten(), params["W2"].flatten()] + [params["b1"], params["b2"]] if self.use_bias else [])
        return torch.cat([self.W1.flatten(), self.W2.flatten()] + [self.b1, self.b2] if self.use_bias else [])

    def get_gradient_vector(self):
        return torch.cat([self.W1.grad.flatten(), self.W2.grad.flatten()] + [self.b1.grad, self.b2.grad] if self.use_bias else [])

    def load_from_vector(self, vector):

        def get_last_elements(v, shape):
            numel = np.prod(shape)
            res = v[-numel:]
            v = v[:-numel]
            return res.reshape(*shape), v

        params = OrderedDict()
        vector = vector.flatten()
        if self.use_bias:
            params["b2"], vector = get_last_elements(vector, [self.e_dim])
            params["b1"], vector = get_last_elements(vector, [self.hidden_size])
        params["W2"], vector = get_last_elements(vector, [self.e_dim, self.hidden_size])
        params["W1"], vector = get_last_elements(vector, [self.hidden_size, self.e_dim])
        assert len(vector) == 0

        return params

    def forward(self, image_features, text_features, weights=None, params=None):
        param_dict = OrderedDict(params) if params is not None else OrderedDict()

        if weights is not None and not self.ignore_passed_weights:
            param_dict.update(self.load_from_vector(weights))

        W1 = get_from_dict_or_default(self, param_dict, "W1")
        W2 = get_from_dict_or_default(self, param_dict, "W2")
        b1 = get_from_dict_or_default(self, param_dict, "b1")
        b2 = get_from_dict_or_default(self, param_dict, "b2")

        normalized_W1 = W1
        normalized_W2 = W2
        if not self.straight_through:
            normalized_W1 = W1/math.sqrt(self.e_dim)
            normalized_W2 = W2/math.sqrt(self.hidden_size)

        identity = image_features
        out = F.linear(image_features, normalized_W1, b1)
        out = F.relu(out)
        out = F.linear(out, normalized_W2, b2)
        out += identity
        adapted_image_features = out / out.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.to(torch.float32)
        logits = 100.0 * adapted_image_features @ torch.transpose(text_features, 1, 2)
        logits = torch.squeeze(logits,1)

        return logits

class EmbeddingModule(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.randn([1, input_dim]))

    def reset(self, embedding=None):
        self.embedding.data *= 0
        if embedding is None:
            embedding = torch.randn_like(self.embedding)
        self.embedding.data += embedding


    def forward(self, params=None):
        param_dict = OrderedDict(params) if params is not None else OrderedDict()
        embedding = get_from_dict_or_default(self, param_dict, "embedding")

        return embedding

class MetaModel(MetaModule):
    def _init(self, mnet=None, hnet=None, enet=None, alpha=1):
        super().__init__()
        self.mnet = mnet
        self.hnet = hnet
        self.enet = enet
        self.alpha=alpha
        if self.hnet is not None:
            if self.enet is not None:
                self.meta_params = list(self.hnet.parameters())+list(self.enet.parameters())
                self.inner_module = self.enet
            else:
                self.meta_params = list(self.hnet.parameters())
                self.inner_module = self.hnet
        else:
            self.meta_params = list(self.mnet.parameters())
            self.inner_module = self.mnet

        self.inner_params=  self._get_inner_params()

    def __init__(self, mnet=None, hnet=None, enet=None, inner_param=None, mainnet_use_bias=None, mainnet_hidden_dim=None, hypernet_hidden_dim=None, embedding_dim=None, straight_through=False, config=None):
        super().__init__()
        if "alpha" not in config:
            alpha=1
        else:
            alpha = config["alpha"]

        if mnet is not None or hnet is not None or enet is not None:
            self._init(mnet, hnet, enet, alpha)
        if inner_param == "enet":
            self.mnet = CLIPAdapter(clip_utils.embedding_size[config["clip_model"]],
                           mainnet_hidden_dim, use_bias=mainnet_use_bias, no_weights=True, straight_through=straight_through)
            self.hnet = HyperGenerator(self.mnet, e_dim=embedding_dim,
                              hidden_dims=hypernet_hidden_dim, )
            self.enet = EmbeddingModule(embedding_dim)
            self.meta_params = list(self.hnet.parameters())+list(self.enet.parameters())
            self.inner_module = self.enet

        elif inner_param == "hnet":
            self.mnet = CLIPAdapter(clip_utils.embedding_size[config["clip_model"]],
                           mainnet_hidden_dim, use_bias=mainnet_use_bias, no_weights=True, straight_through=straight_through)
            self.hnet = HyperGenerator(self.mnet, e_dim=clip_utils.embedding_size[config["clip_model"]],
                              hidden_dims=hypernet_hidden_dim, )
            self.enet = None
            self.meta_params = list(self.hnet.parameters())
            self.inner_module = self.hnet

        elif inner_param == "mnet":
            self.mnet = CLIPAdapter(clip_utils.embedding_size[config["clip_model"]],
                           mainnet_hidden_dim, use_bias=mainnet_use_bias, no_weights=False, straight_through=straight_through)
            self.hnet = None
            self.enet = None
            self.meta_params = list(self.mnet.parameters())
            self.inner_module = self.mnet
        self.alpha=alpha
        self.inner_params=  self._get_inner_params()

    def _get_inner_params(self):
        params = OrderedDict()
        for (name, param) in self.named_parameters():
            if any([id(param) == id(b) for b in self.inner_module.parameters()]):
                params[name] = param
        return params

    def get_inner_params(self):
        return self.inner_params 

    def _forward(self, sample_image_features, sample_text_features, sample_ques_emb, params=None):
        if self.hnet is not None:
            if self.enet is None:
                weights = self.hnet.forward(uncond_input=sample_ques_emb, params=self.get_subdict(params, "hnet"))
            else:
                weights = self.hnet.forward(uncond_input=self.enet(params=self.get_subdict(params, "enet")))
            similarity = self.mnet(sample_image_features, sample_text_features, weights=weights)
        else:
            similarity = self.mnet(sample_image_features, sample_text_features, params=self.get_subdict(params, "mnet"))

        return similarity

    def forward(self, *args,  params=None, **kwargs):
        if self.alpha == 1:
            return self._forward( *args, params=params, **kwargs)
        init_output = self._forward( *args, **kwargs)
        adapted_output = self._forward( *args, params = params, **kwargs)
        return self.alpha * (adapted_output - init_output) + init_output

    def get_mainnet_weights(self, ques_emb = None, params=None):
        if self.hnet is not None:
            if self.enet is None:
                return self.hnet.forward(uncond_input=ques_emb, params=self.get_subdict(params, "hnet"))
            else:
                return self.hnet.forward(uncond_input=self.enet(params=self.get_subdict(params, "enet")))
        else:
            return self.mnet.get_parameter_vector(params=self.get_subdict(params, "mnet"))
