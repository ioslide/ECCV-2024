from copy import deepcopy
from loguru import logger as log
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import torch.optim as optim

__all__ = ["setup"]

# This code is adapted from 
# https://github.com/MotasemAlfarra/Online_Test_Time_Adaptation/blob/main/tta_methods/shot.py

class SHOT(nn.Module):
    """
    SHOT method from https://arxiv.org/abs/2002.08546
    """
    def __init__(self, model, optimizer, cfg):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.steps = cfg.OPTIM.STEPS #Modify this based on optimal parameters
        assert self.steps > 0, "shot requires >= 1 step(s) to forward and update"

        assert cfg.ADAPTER.SHOT.beta_clustering_loss > 0, "beta_clustering_loss must be > 0, otherwise use SHOT"
        self.beta_clustering_loss = cfg.ADAPTER.SHOT.beta_clustering_loss
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
            
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def forward(self, x, **kwargs):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        
        if self.beta_clustering_loss:
            outputs, features = model(x, return_feature=True)
        else:
            outputs = model(x)

        softmax_out = outputs.softmax(1)
        msoftmax = softmax_out.mean(0)

        # ================ SHOT-IM ================
        l_ent = - (softmax_out * torch.log(softmax_out + 1e-5)).sum(-1).mean(0)
        l_div =  (msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
        
        loss = l_ent + l_div
        # ================ SHOT-IM ================
        
        
        # =============== Full SHOT, SHOT-IM + clustering loss ===============
        # normalize features
        features = features / features.norm(dim=1, keepdim=True)
        # Equivalent to the following line in the original implementation
        # https://github.com/tim-learn/SHOT/blob/07d0c713e4882e83fded1aff2a447dff77856d64/digit/uda_digit.py#L386
        # features = (features.t()/torch.norm(features,p=2,dim=1)).t()
        
        # Compute clustering loss
        # Compute centroids of each class            
        K = outputs.shape[1]
        aff = softmax_out
        
        initial_centroids = torch.matmul(aff.t(), features)
        # Equivalent to the following line in the original implementation
        # https://github.com/tim-learn/SHOT/blob/07d0c713e4882e83fded1aff2a447dff77856d64/digit/uda_digit.py#L391
        # initial_centroids = aff.transpose().dot(features)
        
        #normalize centroids
        initial_centroids = initial_centroids / (1e-8 + aff.sum(0, keepdim=True).t())
        # aff.sum(0, keepdim=True).t() is equivalente to aff.sum(0)[:, None]

        # Compute distances to centroids
        distances = torch.cdist(features, initial_centroids, p=2)
        # Compute pseudo labels
        pseudo_labels = distances.argmin(axis=1)
        
        # I don't know why they do this, but it's in the original implementation
        for _ in range(1):
            pseudo_labels = pseudo_labels.to('cpu')
            aff = torch.eye(K)[pseudo_labels].to(aff.device)
            initial_centroids = torch.matmul(aff.t(), features)
            initial_centroids = initial_centroids / (1e-8 + aff.sum(0, keepdim=True).t())
            distances = torch.cdist(features, initial_centroids, p=2)
            pseudo_labels = distances.argmin(axis=1)
            
        # Compute clustering loss
        loss += self.beta_clustering_loss * F.cross_entropy(outputs, pseudo_labels)
        # =============== Full SHOT, SHOT-IM + clustering loss ===============
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs
    

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names
    
def configure_model(model, update_bn_only=True):
    """Configure model for use with shot."""
    # train mode, because shot optimizes the model to minimize entropy
    # SHOT updates all parameters of the feature extractor, excluding the last FC layers
    # Original SHOT implementation
    if not update_bn_only:
        model.train()
        # is this needed? review later
        model.requires_grad_(True)
        # Freeze FC layers
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.requires_grad_(False)
                
    else:
        # In case we want shot to update only the BN layers
        # disable grad, to (re-)enable only what shot updates (originally not used by shot but other papers use it when using shot)
        model.requires_grad_(False)
        # configure norm for shot updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def setup(model, cfg):
    log.info("Setup TTA method: shot")
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
    TTA_model = SHOT(
        model, 
        optimizer,
        cfg,
    )
    return TTA_model