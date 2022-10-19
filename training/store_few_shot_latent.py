from os import path

import numpy as np
import torch

from training.conditional_model_learn import ConditionalModelTraining


class StoreFewShotLatent(ConditionalModelTraining):
    def __init__(self, meta_module, image_features, text_features, ques_emb, config, device, compute_hessian=False,
                 reset_normal_embedding=False, **kwargs):
        assert hasattr(meta_module, "hnet")
        feature_dim = np.sum([v.numel() for v in meta_module.get_inner_params().values()])
        super().__init__(meta_module, None, None, image_features, text_features, ques_emb, config, device, feature_dim,
                         None, None, compute_hessian=compute_hessian, **kwargs)
        self.reset_normal_embedding=reset_normal_embedding

    def reset_task(self, guided_inner):
        if self.reset_normal_embedding:
            self.meta_module.enet.reset() 

    def train_step(self, log_dict, log_file=None):
        clip_embedding=[]
        embedding=[]
        w_vect=[]
        task_idx=[]
        hessian=[]
        coco=[]

        if path.exists(log_file):
            matrix_dict = torch.load(log_file)
            clip_embedding.append(matrix_dict["clip_embedding"])
            coco.append(matrix_dict["coco"])

            if self.base_module.mnet.no_weight:
                embedding.append(matrix_dict["embedding"])
            else:
                w_vect.append(matrix_dict["w_vect"])

            task_idx.append(matrix_dict["task_idx"])
            if self.compute_hessian:
                hessian.append(matrix_dict["hessian"])

        clip_embedding.append(self.ques_batch.unsqueeze(0).cpu())
        coco.append(self.coco_batch.unsqueeze(0).cpu())
        if self.base_module.mnet.no_weight:
            embedding.append(self.net_batch.unsqueeze(0).cpu())
        else:
            w_vect.append(self.net_batch.unsqueeze(0).cpu())

        task_idx.append(self.task_batch.unsqueeze(0).cpu())
        if self.compute_hessian:
            hessian.append(self.hessian_batch.unsqueeze(0).cpu())

        matrix_dict=dict()
        matrix_dict["clip_embedding"]=torch.cat(clip_embedding, dim=0)
        matrix_dict["coco"]=torch.cat(coco, dim=0)

        if self.base_module.mnet.no_weight:
            matrix_dict["embedding"]=torch.cat(embedding, dim=0)
        else:
            matrix_dict["w_vect"]=torch.cat(w_vect, dim=0)

        matrix_dict["task_idx"]=torch.cat(task_idx, dim=0)
        if self.compute_hessian:
            matrix_dict["hessian"]=torch.cat(hessian, dim=0)

        print("Saving new tensor")
        torch.save(matrix_dict, log_file)

    def test_step(self, log_dict):
        pass

    def guided_inner(self, *args, **kwargs):
        pass

    def feed_net_feature(self, inner_params):
        if self.base_module.mnet.no_weight:
            return inner_params["enet.embedding"].detach()
        return self.base_module.get_mainnet_weights(params = inner_params).detach()
