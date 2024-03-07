# paper title: robust weight-aware test-time adaptation for non-stationary scenarios
from copy import deepcopy
from loguru import logger as log
import random
from collections import deque
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
from methods.PRDCL.transformers_cotta import get_tta_transforms
import torch.nn.functional as F
from collections import defaultdict
import math
import numpy as np

__all__ = ["setup"]


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """Consistency loss between two softmax distributions."""
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()

def wise_ft(theta_0, theta_1, alpha, fisher=False, FIM_list=None):
    if not fisher:
        theta = {
            key: (1 - alpha) * theta_0[key].cuda() + alpha * theta_1[key].cuda()
            for key in theta_0.keys()
        }
        unique_keys = set(theta_1.keys()) - set(theta_0.keys())
        for item in unique_keys:
            theta[item] = theta_1[item]
    else:
        FIM_theta_0, FIM_theta_1, final_theta = {}, {}, {}
        for key in theta_0.keys():
            try:
                FIM_theta_0[key] = theta_0[key].cuda() * (FIM_list[0][key] + 1e-8).cuda()
            except:
                pass

        for key in theta_1.keys():
            try:
                FIM_theta_1[key] = theta_1[key].cuda() * (FIM_list[1][key] + 1e-8).cuda()
            except:
                pass
            
        for key in theta_0.keys():
            try:
                a = (
                    (1 - alpha) * FIM_theta_0[key] + alpha * FIM_theta_1[key]
                ).cuda()
                b = (
                    (1 - alpha) * (FIM_list[0][key] + 1e-8) + alpha * (FIM_list[1][key] + 1e-8)
                ).cuda()
                final_theta[key] = (
                    a / b
                )
            except:
                pass

        unique_keys = set(theta_1.keys()) - set(theta_0.keys())
        for item in unique_keys:
            final_theta[item] = FIM_theta_1[item]

    return final_theta

class PRDCL(nn.Module):
    def __init__(self, cfg,model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = cfg.OPTIM.STEPS
        self.cfg = cfg
        assert self.steps > 0, "PRDCL requires >= 1 step(s) to forward and update"

        self.transforms = get_tta_transforms(cfg.CORRUPTION.DATASET)
        self.eps = 1e-8
        self.model_state, self.optimizer_state, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.model_anchor.eval()
        self.FIM_list = deque(maxlen=2)
        self.state_list = deque(maxlen=2)

        self.anchor_state = deepcopy(self.model.state_dict())
        self.anchor_FIM = None

        self.trainable_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}

        self.min_weight, self.max_weight =  1e8, -1e8
        self.index = 0

        self.param_groups = self.optimizer.param_groups


    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    def reweight_theta_by_fisher(self):
        scale = 0.0001
        FIM_t = defaultdict(lambda: 0.0)
        for name, param in self.trainable_dict.items():
            FIM_t[name] += (torch.pow(param.grad, 2))

        for name, param in FIM_t.items():
            FIM_t[name] = torch.min(FIM_t[name], torch.tensor(scale))

        if self.index == 0:
            self.anchor_FIM = FIM_t

        theta = wise_ft(
            self.anchor_state, 
            self.model.state_dict(), 
            alpha=self.cfg.ADAPTER.PRDCL.FISHER_ALPHA,
            fisher=True,
            FIM_list=[
                self.anchor_FIM, 
                FIM_t
            ]
        )
        self.model.load_state_dict(theta, strict=False)


    @torch.enable_grad()
    def forward_and_adapt(self,x):
        self.optimizer.zero_grad()
        # ============forward============
        logits = self.model(x)
        aug_logits = self.model(self.transforms(x))

        with torch.no_grad():
            anchor_logits = self.model_anchor(x)

        # ============backward============
        loss_1_0 = softmax_entropy(logits).mean(0)
        loss_1_1 = consistency(logits, aug_logits).mean(0)
        loss_1_2 = consistency(logits, anchor_logits).mean(0)

        loss_1 = loss_1_0 + self.cfg.ADAPTER.PRDCL.CONSISTENCY * loss_1_1 + self.cfg.ADAPTER.PRDCL.CONSISTENCY2 * loss_1_2
        loss_1.backward()

        self.reweight_theta_by_fisher()
        self.optimizer.step()

        self.index +=1
        return logits

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,self.model_state, self.optimizer_state)


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())

    for param in model_anchor.parameters():
        param.detach_()
    return model_state, optimizer_state, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

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
    """Check model for compatability with PRDCL."""
    is_training = model.training
    assert is_training, "PRDCL needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "PRDCL needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "PRDCL should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "PRDCL needs normalization for its optimization"

def setup(model, cfg):
    log.info("Setup TTA method: PRDCL")
    model = configure_model(model)
    params, param_names = collect_params(model)
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            params, 
            lr=float(cfg.OPTIM.LR),
            momentum=float(cfg.OPTIM.MOMENTUM),
            weight_decay=float(cfg.OPTIM.WD),
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(
            params, 
            lr=float(cfg.OPTIM.LR),
            weight_decay=float(cfg.OPTIM.WD)
        )
    TTA_model = PRDCL(
        cfg,
        model, 
        optimizer
    )
    TTA_model.reset()
    return TTA_model