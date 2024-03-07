from copy import deepcopy
from loguru import logger as log
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim

__all__ = ["setup"]

# This code is adapted from the official implementation:
# https://github.com/DequanWang/tent/blob/master/tent.py

class TENT(nn.Module):
    def __init__(self, cfg,model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, **kwargs):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    @torch.enable_grad() 
    def forward_and_adapt(self,x):
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        return outputs

    def reset(self): # sourcery skip: raise-specific-error
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    check_model(model)
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def check_model(model):
    is_training = model.training
    assert is_training, "Tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "Tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "Tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "Tent needs normalization for its optimization"

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def setup(model, cfg):
    log.info("Setup TTA method: Tent")
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
    TTA_model = TENT(
        cfg,
        model, 
        optimizer
    )
    return TTA_model