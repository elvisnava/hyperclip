# Credit: modification of code from https://github.com/AndreyGuzhov/AudioCLIP

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from clip.model import CLIP

from model.custom_hnet import Mlp

ClipFeatures = Tuple[
    Optional[torch.Tensor],  # text
    Optional[torch.Tensor],  # image
    Optional[torch.Tensor]   # hyper
]


ClipLogits = Tuple[
    Optional[torch.Tensor],  # hyper x image
    Optional[torch.Tensor],  # hyper x text
    Optional[torch.Tensor]   # image x text
]


ClipOutput = Tuple[
    Tuple[ClipFeatures, ClipLogits],
    Optional[torch.Tensor]   # loss
]

class HyperCLIP(CLIP):

    def __init__(self,
                 embed_dim: int = 1024,
                 # vision
                 image_resolution: int = 224,
                 vision_layers: Union[Tuple[int, int, int, int], int] = (3, 4, 6, 3),
                 vision_width: int = 64,
                 vision_patch_size: Optional[int] = None,
                 # text
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 12,
                 # hyper
                 hyper_model: str = "mlp", # choose between "mlp" and "embedder_hypernet"
                 mainnet_param_count: int = 1024*256*2,
                 hyper_hidden_dims: List[int] = [512],
                 # pretrained model
                 pretrained_it_location: Optional[str] = None,
                 pretrained_hyper_location: Optional[str] = None):

        super(HyperCLIP, self).__init__(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers
        )

        self.embed_dim = embed_dim
        self.hyper_model = hyper_model

        self.pretrained_it_location = pretrained_it_location
        self.pretrained_hyper_location = pretrained_hyper_location

        self.logit_scale_hi = torch.nn.Parameter(torch.log(torch.ones([]) * 100))
        self.logit_scale_ht = torch.nn.Parameter(torch.log(torch.ones([]) * 100))

        if pretrained_it_location is not None:
            self.load_state_dict(torch.jit.load(self.pretrained_it_location, map_location='cpu').state_dict(), strict=False)
            print('Image & Text weights loaded')

        if self.hyper_model == "mlp":
            self.hyper = Mlp(mainnet_param_count, hyper_hidden_dims, embed_dim, 'softplus')
        else:
            raise ValueError(f"Unsupported hyper model {self.hyper_model}")

        if pretrained_hyper_location is not None:
            self.hyper.load_state_dict(torch.load(self.pretrained_hyper_location, map_location='cpu'), strict=False)
            print('Hyper weights loaded')

    @property
    def device(self):
        return self.visual.conv1.weight.device

    def encode_hyper(self, weights: torch.Tensor) -> torch.Tensor:
        return self.hyper(weights.to(self.device))

    def forward(self,
                hyper: Optional[torch.Tensor] = None,
                image: Optional[torch.Tensor] = None,
                text: Optional[Union[List[List[str]],torch.Tensor]] = None,
                batch_indices: Optional[torch.Tensor] = None,
                # precomputed embeddings
                precomputed_it_embs: bool = False) -> ClipOutput:

        hyper_features = None
        image_features = None
        text_features = None
        sample_weights = None

        if hyper is not None:
            hyper_features = self.encode_hyper(hyper)
            hyper_features = hyper_features / hyper_features.norm(dim=-1, keepdim=True)

        if image is not None:
            if precomputed_it_embs:
                image_features = image
            else:
                image_features = self.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if text is not None:
            if precomputed_it_embs:
                text_features = text
            else:
                if batch_indices is None:
                    batch_indices = torch.arange(len(text), dtype=torch.int64, device=self.device)
                text_features = self.encode_text(text, '{}', batch_indices)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if hasattr(self, 'class_weights') and hasattr(self, 'label_to_class_idx'):
                sample_weights = torch.stack([
                    sum(self.class_weights[self.label_to_class_idx[label]] for label in entities)
                    for idx, entities in enumerate(text) if idx in batch_indices
                ])

        features: ClipFeatures = (hyper_features, image_features, text_features)

        logit_scale_hi = torch.clamp(self.logit_scale_hi.exp(), min=1.0, max=100.0)
        logit_scale_ht = torch.clamp(self.logit_scale_ht.exp(), min=1.0, max=100.0)
        logit_scale_it = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)

        logits_hyper_image = None
        logits_hyper_text = None
        logits_image_text = None

        if (hyper_features is not None) and (image_features is not None):
            logits_hyper_image = logit_scale_hi * hyper_features @ image_features.T

        if (hyper_features is not None) and (text_features is not None):
            logits_hyper_text = logit_scale_ht * hyper_features @ text_features.T

        if (image_features is not None) and (text_features is not None):
            logits_image_text = logit_scale_it * image_features @ text_features.T

        logits: ClipLogits = (logits_hyper_image, logits_hyper_text, logits_image_text)

        loss = self.loss_fn(logits, sample_weights)

        return (features, logits), loss

    def loss_fn(self, logits: ClipLogits, sample_weights: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        logits_hyper_image, logits_hyper_text, logits_image_text = logits

        if logits_hyper_image is not None:
            batch_size = logits_hyper_image.shape[0]
        elif logits_hyper_text is not None:
            batch_size = logits_hyper_text.shape[0]
        elif logits_image_text is not None:
            batch_size = logits_image_text.shape[0]
        else:
            return None

        reference = torch.arange(
            batch_size,
            dtype=torch.int64,
            device=self.device
        )

        loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        num_modalities: int = 0
        scale = torch.tensor(1.0, dtype=self.dtype, device=self.device)

        if logits_hyper_image is not None:
            loss_hi = F.cross_entropy(
                logits_hyper_image, reference, weight=sample_weights
            ) + F.cross_entropy(
                logits_hyper_image.transpose(-1, -2), reference, weight=sample_weights
            )
            loss = loss + loss_hi
            num_modalities += 1

        if logits_hyper_text is not None:
            loss_ht = F.cross_entropy(
                logits_hyper_text, reference, weight=sample_weights
            ) + F.cross_entropy(
                logits_hyper_text.transpose(-1, -2), reference, weight=sample_weights
            )
            loss = loss + loss_ht
            num_modalities += 1

        if logits_image_text is not None:
            loss_it = F.cross_entropy(
                logits_image_text, reference, weight=sample_weights
            ) + F.cross_entropy(
                logits_image_text.transpose(-1, -2), reference, weight=sample_weights
            )
            loss = loss + loss_it
            num_modalities += 1

        for idx in range(num_modalities):
            scale = scale * (idx + 1)

        return loss / scale

    @property
    def loss_fn_name(self) -> str:
        return 'Cross Entropy'


def build_hyperclip_from_classic_clip(state_dict: Union[dict, str],
                                      hyper_model: str = "mlp",
                                      mainnet_param_count: int = 2014*256*2,
                                      hyper_hidden_dims: List[int] = [512],
                                      pretrained_it_location: Optional[str] = None,
                                      pretrained_hyper_location: Optional[str] = None):

    if isinstance(state_dict, str):
        state_dict = torch.jit.load(state_dict, map_location='cpu').state_dict()

    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = HyperCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        hyper_model, mainnet_param_count, hyper_hidden_dims, pretrained_it_location, pretrained_hyper_location
    )

    return model
