# Meta-Learning via Classifier(-free) Diffusion Guidance

[arxiv](https://arxiv.org/abs/2210.08942) | [BibTeX](#citation)

**Meta-Learning via Classifier(-free) Diffusion Guidance**<br/>
[Elvis Nava](https://github.com/elvisnava)\*,
[Seijin Kobayashi](https://github.com/seijin-kobayashi)\*,
Yifei Yin,
Robert K. Katzschmann,
Benjamin F. Grewe<br/>
\* equal contribution

# Installation

The `hyperclip` conda environment can be created with the following commands:
```
conda env create -f environment.yml
conda activate hyperclip
pip install git+https://github.com/openai/CLIP.git
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install -e .
```

To setup Weights and Biases run
```
wandb login
```
and paste your W&B API key.

# Meta-VQA Dataset

To re-compute the Meta-VQA dataset, first download the [original VQA v2 dataset](https://visualqa.org/download.html) and place it in the `data/VQA/` folder, and then run (while in the `hyperclip` environment):
```
python scripts/precompute_image_features.py
python scripts/precompute_ques_features.py
python scripts/precompute_text_features.py
```
to re-generate the pre-computed CLIP embeddings for images, task questions and answers.

# Experiment scripts

To train multitask/MAML baselines or an unconditional Hypernetwork generative model (to later use as basis for conditional generation), use the script:
```
python scripts/train_few_shot.py [...]
```

To train a number of our models, we first need to prepare a precomputed "dataset" of fine-tuned networks/hnet latents/vae latents. We can do so with the script:
```
python scripts/precompute_adaptation.py (--few_shot_checkpoint <wandb id of train_few_shot.py hnet run> | --vae_checkpoint <wandb id of train_vae.py run>) [...]
```

In order to train the unconditional VAE hypernetwork (alternative to the previous HNET as basis for conditional generation methods), use the script:
```
python scripts/train_vae.py --precompute_checkpoint <wandb id of precompute_adaptation.py run> [...]
```

To train the HyperCLIP encoder (either from precomputed VAE/HNET fine-tunings, a VAE, or an HNET), use the script:
```
python scripts/train_hyperclip.py (--precompute_checkpoint <wandb id of precompute_adaptation.py run> | --vae_checkpoint <wandb id of train_vae.py run> | --few_shot_checkpoint <wandb id of train_few_shot.py run>) [...]
```

To train a hypernetwork latent diffusion model (HyperLDM), use the script:
```
python scripts/train_latent_diffusion.py (--precompute_checkpoint <wandb id of precompute_adaptation.py run> | --vae_checkpoint <wandb id of train_vae.py run> | --few_shot_checkpoint <wandb id of train_few_shot.py>) [...]
```


# Citation
```
@misc{nava_meta-learning_2022,
	title = {Meta-{Learning} via {Classifier}(-free) {Diffusion} {Guidance}},
	url = {http://arxiv.org/abs/2210.08942},
	doi = {10.48550/arXiv.2210.08942},
	publisher = {arXiv},
	author = {Nava, Elvis and Kobayashi, Seijin and Yin, Yifei and Katzschmann, Robert K. and Grewe, Benjamin F.},
	month = oct,
	year = {2022},
	note = {arXiv:2210.08942 [cs]},
	keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences},
    copyright = {arXiv.org perpetual, non-exclusive license}
}
```
