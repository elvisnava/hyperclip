import os

import torch
import wandb
from model.custom_hnet import (CLIPAdapter, EmbeddingModule, HyperEncoder,
                               HyperGenerator, MetaModel)
from training.vae_learn import sample_from_enc

base_path = os.path.dirname(os.path.dirname(__file__))

def load_few_shot_to_metamodel(run, device):
    loaded_config = run.config
    meta_module = MetaModel(
       inner_param=loaded_config["inner_param"],
       mainnet_use_bias=loaded_config["mainnet_use_bias"],
       mainnet_hidden_dim=loaded_config["mainnet_hidden_dim"],
       hypernet_hidden_dim=[] if loaded_config["hypernet_hidden_dim"]=="" else [int(i) for i in loaded_config["hypernet_hidden_dim"].split(",")],
       embedding_dim=loaded_config["embedding_dim"],
       straight_through=loaded_config["straight_through"],
       config=loaded_config).to(device)
    loaded_model_path = base_path + "/evaluation/few_shot/meta_module_" + str(run.name) + ".pth"
    meta_module.load_state_dict(torch.load(loaded_model_path), strict=False)
    return meta_module

def load_vae(run, device):
    config=run.config

    api = wandb.Api()
    if "precompute_checkpoint" in config:
        print("Loading from precomputed run {}".format(config["precompute_checkpoint"]))
        precomp_run = api.run(config["precompute_checkpoint"])
        precomp_config = precomp_run.config
        config.update({"few_shot_checkpoint": precomp_config["few_shot_checkpoint"]}, allow_val_change=True)

    loaded_run = api.run(config["few_shot_checkpoint"])
    tmp_meta_module = load_few_shot_to_metamodel(loaded_run, device)

    hidden_dims = [int(h) for h in config["vae_hidden_dim"].split(",")]
    hnet_enc = HyperEncoder(tmp_meta_module.mnet, e_dim=config["vae_noise_dim"],
                                hidden_dims=hidden_dims, normalize="normalize" in config and config["normalize"]).to(device)
    hidden_dims.reverse()
    hnet_gen = HyperGenerator(tmp_meta_module.mnet, e_dim=config["vae_noise_dim"],
                              hidden_dims=hidden_dims, normalize="normalize" in config and config["normalize"]).to(device)

    loaded_model_path = base_path + "/evaluation/vae/hnet_gen_" + str(run.name) + ".pth"
    hnet_gen.load_state_dict(torch.load(loaded_model_path), strict=False)
    loaded_model_path = base_path + "/evaluation/vae/hnet_enc_" + str(run.name) + ".pth"
    hnet_enc.load_state_dict(torch.load(loaded_model_path), strict=False)

    return hnet_gen, hnet_enc, tmp_meta_module

def load_vae_to_metamodel(run, device):
    hnet_gen, hnet_enc, tmp_meta_module = load_vae(run, device)

    embedding = sample_from_enc(hnet_enc, tmp_meta_module.get_mainnet_weights(params = None).detach()).detach()
    enet = EmbeddingModule(hnet_gen.input_dim).to(device)
    enet.reset(embedding=embedding)

    mnet = CLIPAdapter(e_dim=tmp_meta_module.mnet.e_dim,
                        hidden_layers=[tmp_meta_module.mnet.hidden_size],
                        use_bias=tmp_meta_module.mnet.use_bias,
                        straight_through=tmp_meta_module.mnet.straight_through,
                        no_weights=True).to(device)
    meta_module = MetaModel(mnet=mnet, hnet=hnet_gen, enet=enet, config=run.config)

    return meta_module

def load_metamodel_from_checkpoint(config, device):
    precomputed_latent = None
    precomputed_latent_train_eval = None
    precomputed_latent_val_eval = None
    api = wandb.Api()
    if "precompute_checkpoint" in config:
        print("Loading from precomputed run {}".format(config["precompute_checkpoint"]))
        precomp_run = api.run(config["precompute_checkpoint"])
        precomp_config = precomp_run.config
        config.update({"few_shot_checkpoint": None if "few_shot_checkpoint" not in precomp_config else precomp_config["few_shot_checkpoint"]}, allow_val_change=True)
        config.update({"vae_checkpoint": None if "vae_checkpoint" not in precomp_config else precomp_config["vae_checkpoint"]}, allow_val_change=True)

        precomputed_file = base_path + "/evaluation/precompute_adaptation/" + str(precomp_run.name) + ".pth"
        precomputed_latent = torch.load(precomputed_file) #{k:v.to(device) for (k,v) in torch.load(precomputed_file).items()}

        try:
            precomputed_file_train_eval = base_path + "/evaluation/precompute_adaptation/" + str(precomp_run.name) + "_train_eval.pth"
            precomputed_file_val_eval = base_path + "/evaluation/precompute_adaptation/" + str(precomp_run.name) + "_val_eval.pth"
            precomputed_latent_train_eval = {k:v.to(device) for (k,v) in torch.load(precomputed_file_train_eval).items()}
            precomputed_latent_val_eval = {k:v.to(device) for (k,v) in torch.load(precomputed_file_val_eval).items()}
        except:
            print("Did not find eval latent")

    if "few_shot_checkpoint" in config and config["few_shot_checkpoint"] is not None:
        print("Creating MetaModule from FewShot trained model")
        loaded_run = api.run(config["few_shot_checkpoint"])
        meta_module = load_few_shot_to_metamodel(loaded_run, device)
    elif "vae_checkpoint" in config and config["vae_checkpoint"] is not None:
        print("Creating MetaModule from VAE model")
        loaded_run = api.run(config["vae_checkpoint"])
        meta_module = load_vae_to_metamodel(loaded_run, device)
    else:
        return NotImplementedError("something's wrong")

    return meta_module, precomputed_latent, precomputed_latent_train_eval, precomputed_latent_val_eval

def load_vae_and_metamodel_from_checkpoint(config, device):
    api = wandb.Api()
    assert "vae_checkpoint" in config and config["vae_checkpoint"] is not None
    loaded_run = api.run(config["vae_checkpoint"])
    meta_module, precomputed_latent, precomputed_latent_train_eval, precomputed_latent_val_eval = \
        load_metamodel_from_checkpoint(loaded_run.config, device)
    hnet_gen, hnet_enc, _ = load_vae(loaded_run, device)

    return meta_module, hnet_gen, hnet_enc, precomputed_latent, precomputed_latent_train_eval, precomputed_latent_val_eval
