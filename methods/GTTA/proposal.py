from copy import deepcopy
from loguru import logger as log
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import os
__all__ = ["setup"]
import logging

import os
import tqdm
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from core.model.build import split_up_model
from core.data.data_loading import get_source_loader

from methods.GTTA.style_transfer import TransferNet



def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class GTTA_model(nn.Module):
    def __init__(self, cfg,model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = 1
        self.dataset_name = cfg.CORRUPTION.DATASET

        batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
        _, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                               batch_size=batch_size_src, ckpt_path=cfg.CKPT_PATH,
                                               percentage=cfg.ADAPTER.GTTA.SOURCE_PERCENTAGE,
                                               workers=2)
        self.steps_adain = cfg.ADAPTER.GTTA.STEPS_ADAIN
        self.use_style_transfer = cfg.ADAPTER.GTTA.USE_STYLE_TRANSFER
        self.lam = cfg.ADAPTER.GTTA.LAMBDA_MIXUP
        self.buffer_size = 100000
        self.counter = 0
        ckpt_dir = cfg.CKPT_DIR
        ckpt_path = cfg.CKPT_PATH
        self.device = torch.device("cuda")
        self.avg_conf = torch.tensor(0.9).cuda()
        self.ignore_label = -1
        # Create style-transfer network
        if self.use_style_transfer:
            fname = os.path.join(ckpt_dir, "adain", f"decoder_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth" if self.dataset_name == "domainnet126" else f"decoder_{self.dataset_name}.pth")
            self.adain_model = TransferNet(ckpt_path_vgg=os.path.join(ckpt_dir, "adain", "vgg_normalized.pth"),
                                           ckpt_path_dec=fname,
                                           data_loader=self.src_loader,
                                           num_iters_pretrain=cfg.ADAPTER.GTTA.PRETRAIN_STEPS_ADAIN).to(self.device)
            self.moments_list = [[torch.tensor([], device=self.device), torch.tensor([], device=self.device)] for _ in range(2)]
            self.models = [self.model, self.adain_model]
        else:
            self.adain_model = None
            self.moments_list = None
            self.models = [self.model]

        self.src_loader_iter = iter(self.src_loader)

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)

        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):

        with torch.no_grad():
            outputs_test = self.model(x)

        if self.counter == 0:
            self.filtered_pseudos = self.create_pseudo_labels(outputs_test)
            if self.use_style_transfer:
                self.adain_model.train()
                self.extract_moments(x)

                # Train adain model
                for _ in range(self.steps_adain):
                    # sample source batch
                    try:
                        batch = next(self.src_loader_iter)
                    except StopIteration:
                        self.src_loader_iter = iter(self.src_loader)
                        batch = next(self.src_loader_iter)

                    # train on source data
                    imgs_src = batch[0].to(self.device)

                    self.adain_model.opt_adain_dec.zero_grad()
                    _, loss_content, loss_style = self.adain_model(imgs_src, moments_list=self.moments_list)
                    loss_adain = 1.0 * loss_content + 0.1 * loss_style
                    loss_adain.backward()
                    self.adain_model.opt_adain_dec.step()

        # Train classification model
        with torch.no_grad():
            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            # train on labeled source data
            imgs_src, labels_src = batch[0].to(self.device), batch[1].to(self.device).long()

            if self.use_style_transfer:
                # Generate style transferred images from source images
                imgs_src, _, _ = self.adain_model(imgs_src, moments_list=self.moments_list)
            else:
                # Perform mixup
                batch_size = x.shape[0]
                imgs_src = imgs_src[:batch_size]
                labels_src = labels_src[:batch_size]
                outputs_src = self.model(imgs_src)
                _, ids = torch.max(torch.matmul(outputs_src.softmax(1), outputs_test.softmax(1).T), dim=1)
                imgs_src = self.mixup_data(imgs_src, x[ids], lam=self.lam)

        loss_source = F.cross_entropy(input=self.model(imgs_src), target=labels_src)
        loss_source.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        outputs_test = self.model(x)
        loss_target = F.cross_entropy(input=outputs_test, target=self.filtered_pseudos, ignore_index=-1)
        loss_target.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.counter += 1
        self.counter %= self.steps
        return outputs_test

    @torch.no_grad()
    def mixup_data(self, x_source, x_target, lam=0.25):
        mixed_x = lam * x_target + (1 - lam) * x_source
        return mixed_x

    @torch.no_grad()
    def create_pseudo_labels(self, outputs_test):
        # Create pseudo-labels
        confidences, pseudo_labels = torch.max(outputs_test.softmax(dim=1), dim=1)

        momentum = 0.9
        self.avg_conf = momentum * self.avg_conf + (1 - momentum) * confidences.mean()
        mask = torch.where(confidences < torch.sqrt(self.avg_conf))

        filtered_pseudos = pseudo_labels.clone()
        filtered_pseudos[mask] = self.ignore_label

        return filtered_pseudos

    @torch.no_grad()
    def extract_moments(self, x):
        # Extract image-wise moments from current test batch
        adain_moments = self.adain_model(x)

        # Save moments in a buffer list
        for i_adain_layer, (means, stds) in enumerate(adain_moments):  # Iterate through the adain layers
            self.moments_list[i_adain_layer][0] = torch.cat([self.moments_list[i_adain_layer][0], means], dim=0)
            self.moments_list[i_adain_layer][1] = torch.cat([self.moments_list[i_adain_layer][1], stds], dim=0)
            moments_size = len(self.moments_list[i_adain_layer][0])
            if moments_size > self.buffer_size:
                self.moments_list[i_adain_layer][0] = self.moments_list[i_adain_layer][0][moments_size - self.buffer_size:]
                self.moments_list[i_adain_layer][1] = self.moments_list[i_adain_layer][1][moments_size - self.buffer_size:]

    def reset(self):
        self.moments_list = [[torch.tensor([], device="cuda"), torch.tensor([], device="cuda")] for _ in range(2)]

        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")

        load_model_and_optimizer(self.model, self.optimizer,self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

def collect_params(model):
    """Collect all trainable parameters.
    Walk the model's modules and collect all parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if np in ['weight', 'bias'] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names



def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model



def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    # check_model(model)
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def setup(model, cfg):
    log.info("Setup TTA method: GTTA")
    model = configure_model(model)
    params, param_names = collect_params(model)
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            params, 
            lr=float(cfg.OPTIM.LR),
            dampening=cfg.OPTIM.DAMPENING,
            momentum=float(cfg.OPTIM.MOMENTUM),
            weight_decay=float(cfg.OPTIM.WD),
            nesterov=cfg.OPTIM.NESTEROV
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(
            params, 
            lr=float(cfg.OPTIM.LR),
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=float(cfg.OPTIM.WD)
        )
    TTA_model = GTTA_model(
        cfg,
        model,
        optimizer
    )
    return TTA_model
