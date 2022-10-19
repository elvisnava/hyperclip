import copy
import math
from collections import OrderedDict

import numpy as np
import torch
import wandb
from data.dataloader.clip_vqa import CLIP_VQA
from data.dataloader.coco_tasks import COCO_Tasks
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc_utils import append_dict, mean_dict
from utils.train_utils import get_pred, log_metric, test_accuracy


class GradientBuffer():
    def __init__(self, param_list):
        self.param_list=param_list
        self.reset_buffer()

    def reset_buffer(self):
        self.grad_list=[torch.zeros_like(param) for param in self.param_list]
        self.num = 0

    def accumulate(self):
        for param, grad in zip(self.param_list, self.grad_list):
            if param.grad is not None:
                grad += param.grad.data
        self.num += 1

    def unload(self):
        for param, grad in zip(self.param_list, self.grad_list):
            if param.grad is not None:
                param.grad.data += grad/self.num
        self.reset_buffer()

class MAML():
    def __init__(self, meta_module, meta_optimizer, image_features, text_features, ques_emb, config, coco_categories=None, coco_answer_features=None,
                 extend_coco_size=10 * 870,  # size of virtual extended coco dataset
                 ):
        self.meta_module=meta_module
        self.meta_optimizer=meta_optimizer
        self.image_features=image_features
        self.text_features=text_features
        self.ques_emb=ques_emb
        self.config=config
        self.coco_categories=coco_categories
        self.coco_answer_features=coco_answer_features
        self.extend_coco_size=extend_coco_size
        self.reset_coco()

    def reset(self, batch_size, device):
        self.buffer = GradientBuffer(self.meta_module.meta_params)
        self.batch_size=batch_size
        self.iter=0

    def reset_coco(self):
        self.coco_iter = 0
        self.shuffled_coco_tasks = torch.randperm(self.extend_coco_size)

    def train(self, test_dataloader, inner_params):
        # Validation loss
        log_dict=dict()
        self.meta_module.zero_grad()
        outputs, labels = get_pred(self.meta_module, test_dataloader, params=inner_params)
        loss=F.cross_entropy(outputs, labels)
        loss.backward()
        self.buffer.accumulate()

        if (self.iter % self.batch_size == 0):
            self.buffer.unload()
            self.meta_optimizer.step()
            if self.config["meta_grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(self.meta_module.parameters(), self.config["meta_grad_clip"])

            def get_gradnorm(module):
                return np.sqrt(np.sum([p.grad.pow(2).sum().item() for p in module.parameters() if p.grad is not None])) if module is not None else -1

            log_dict["gradnorm_mnet"] = get_gradnorm(self.meta_module.mnet)
            log_dict["gradnorm_hnet"] = get_gradnorm(self.meta_module.hnet)
            log_dict["gradnorm_enet"] = get_gradnorm(self.meta_module.enet)
        self.iter = self.iter+1

        return log_dict

    def run_epoch(self, data, inner_epochs, inner_lr, meta_batch_size=1, train=False, second_order=False,
                  train_subtype="train", val_subtype="test", keep_tasks_frac=1., extend_coco=False,
                  extend_coco_frac_train=0.5, # frac of tasks to replace with extended coco
                  debug=False, device=None, filter_tasks_by_max_k=None,  filter_tasks_answers=None, n_shot_training=None, epoch=0):

        tasks = list(data.keys())

        if inner_epochs is not None:
            inner_epochs_range = [inner_epochs] if type(inner_epochs) == int else [int(i) for i in inner_epochs.split(",")]
        else:
            inner_epochs_range = None

        if train:
            self.reset(meta_batch_size, None)
            if extend_coco and self.coco_iter >= self.extend_coco_size:
                self.reset_coco()

        log_dict = dict()

        tasks_idxs = np.arange(len(tasks))
        if filter_tasks_by_max_k is not None:
            if not filter_tasks_answers:
                tasks_idxs = [i for i, t in enumerate(tasks) if np.min(np.unique([a for [_,a] in data[t][train_subtype]], return_counts=True)[1]) >= filter_tasks_by_max_k]
            else:
                data = copy.deepcopy(data)
                for t in tasks:
                    ans, count = np.unique([a for [_,a] in data[t][train_subtype]], return_counts=True)
                    filtered_ans = ans[count >= filter_tasks_by_max_k]
                    data[t][train_subtype] = [d for d in data[t][train_subtype] if d[1] in filtered_ans]
                    data[t][val_subtype] = [d for d in data[t][val_subtype] if d[1] in filtered_ans]
                tasks_idxs = [i for i, t in enumerate(tasks) if len(np.unique([a for [_,a] in data[t][train_subtype]])) >= 2]

        tasks = list(data.keys())
        if meta_batch_size > len(tasks_idxs)*keep_tasks_frac:
            meta_batch_size = int(len(tasks_idxs)*keep_tasks_frac)
            print("Warning: batch size too big, decreasing to {}".format(meta_batch_size))

        shuffled_train_tasks = [tasks_idxs[idx] for idx in torch.randperm(len(tasks_idxs))]
        shuffle_for_extended_coco_replace = torch.randperm(len(tasks_idxs))

        for inner_train_iter in tqdm(range(len(tasks_idxs))):
            curr_log_dict = dict()
            enable_coco = extend_coco and shuffle_for_extended_coco_replace[inner_train_iter] < extend_coco_frac_train * len(tasks_idxs)
            if enable_coco:
                train_dataset = COCO_Tasks(categories=self.coco_categories,
                                           dataSubType=train_subtype,
                                           image_features=self.image_features,
                                           coco_answer_features=self.coco_answer_features,
                                           task_seed=self.shuffled_coco_tasks[self.coco_iter])
                test_dataset = COCO_Tasks(categories=self.coco_categories,
                                          dataSubType=val_subtype,
                                          image_features=self.image_features,
                                          coco_answer_features=self.coco_answer_features,
                                          task_seed=self.shuffled_coco_tasks[self.coco_iter])
            else:
                task_idx = shuffled_train_tasks[inner_train_iter]
                if task_idx > len(tasks)*keep_tasks_frac:
                    continue

                train_idx=None
                val_idx=None
                if val_subtype == "random":
                    num_data = len(data[tasks[task_idx]]["train"] + data[tasks[task_idx]]["test"])
                    randfrac = 2/3
                    randperm = np.random.permutation(num_data)
                    val_idx = randperm[:math.floor(num_data*randfrac)]
                    train_idx = randperm[math.floor(num_data*randfrac):]

                train_dataset = CLIP_VQA(meta_data=data,
                                        dataSubType=train_subtype,
                                        task=tasks[task_idx],
                                        image_features=self.image_features,
                                        text_features=self.text_features,
                                        ques_emb=self.ques_emb,
                                        data_idx=train_idx,
                                        n_shot=n_shot_training)
                test_dataset = CLIP_VQA(meta_data=data,
                                        dataSubType=val_subtype,
                                        task=tasks[task_idx],
                                        image_features=self.image_features,
                                        text_features=self.text_features,
                                        ques_emb=self.ques_emb,
                                        data_idx=val_idx
                                        )

            train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

            inner_params = self.meta_module.get_inner_params()

            task_embedding = iter(train_dataloader).next()["ques_emb"][0].to(device).detach()
            if self.config["use_clip_embedding_init"]:
                assert self.meta_module.hnet is not None
                inner_params["enet.embedding"] = task_embedding.requires_grad_()

            train_start_acc, train_start_loss = test_accuracy(self.meta_module, train_dataloader, params=inner_params)
            val_start_acc, val_start_loss = test_accuracy(self.meta_module, test_dataloader, params=inner_params)

            if inner_epochs_range is not None:
                inner_epochs_sampled = inner_epochs_range[0] if len(inner_epochs_range) == 1 else \
                    np.random.randint(inner_epochs_range[0], inner_epochs_range[1]+1)
            else:
                inner_epochs_sampled = None

            # Inner loop
            for _ in range(inner_epochs_sampled):
                outputs, labels = get_pred(self.meta_module, train_dataloader, params=inner_params)
                inner_loss = F.cross_entropy(outputs, labels)

                if debug and inner_train_iter % meta_batch_size == 0:
                    wandb.log({"debug_inner_loss": inner_loss.item()})
                grads = torch.autograd.grad(inner_loss, inner_params.values(), retain_graph=True,
                                            create_graph=True if train and second_order else False)
                params_next = OrderedDict()
                for (name, param), grad in zip(list(inner_params.items()), grads):
                    params_next[name] = param - inner_lr * grad
                inner_params = params_next

            # Train set accuracy
            train_end_acc, train_end_loss = test_accuracy(self.meta_module, train_dataloader, params=inner_params)
            val_end_acc, val_end_loss = test_accuracy(self.meta_module, test_dataloader, params=inner_params)

            curr_log_dict["query_accuracy_start"] = val_start_acc
            curr_log_dict["query_accuracy_end"] = val_end_acc
            curr_log_dict["support_accuracy_start"] = train_start_acc
            curr_log_dict["support_accuracy_end"] = train_end_acc
            curr_log_dict["query_loss_start"] = val_start_loss
            curr_log_dict["query_loss_end"] = val_end_loss
            curr_log_dict["support_loss_start"] = train_start_loss
            curr_log_dict["support_loss_end"] = train_end_loss

            if train:
                curr_log_dict.update(self.train(test_dataloader, inner_params))

            append_dict(log_dict, curr_log_dict)

            if debug and inner_train_iter % meta_batch_size == 0:
                log_metric(mean_dict(log_dict), prefix = "debug_")
                log_dict=dict()

            if enable_coco:
                self.coco_iter += 1

        output_dict = mean_dict(log_dict)
        output_dict["epoch"] = epoch

        return output_dict
