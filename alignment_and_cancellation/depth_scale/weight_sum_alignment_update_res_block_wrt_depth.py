import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np 

from scion import Scion, Spectral

import matplotlib.pyplot as plt
import sys
from dataclasses import dataclass

from decimal import *
getcontext().prec = 32

@dataclass
class Hyperparameters:
    # data hyperparams
    batch_size : int = 1024
    n_trials : int = 5
    apply_Z : bool = True
    metric_for_Z : bool = False
    vector : str = "h"
    net : str = "RESNET" # "RESNET", "MLP"
    act : str = 'relu' # 'relu',  'tanh'
    target_ell : int = 3
    alpha : float = 1.0
    lr : float = 1e-2
    momentum : float = 0.9
    unconstrained : bool = True


from datargs import parse
args = parse(Hyperparameters)

print(args)

torch.set_default_dtype(torch.float64)


if args.metric_for_Z and not args.apply_Z:
    assert False, f"cannot compute the metric for Z if Z is not applied" 


class RESNET(nn.Module):
    """
    Variable-depth ResNet using paper notation (ℓ is 1-indexed):
      W_1:       Linear(n0, d),   h_1(x) = W_1 x,                       h'_1(x) = ϕ(h_1(x))
      W_2..W_L-1 Linear(d, d),    h_ℓ(x) = h_{ℓ-1}(x) + ϕ(W_ℓ h'_{ℓ-1}(x)),  h'_ℓ(x) = ϕ(h_ℓ(x))
      W_L:       Linear(d, n3),   h_L(x) = W_L h'_{L-1}(x)  (output, no activation)

    Args:
        n0 (int): input dimension
        d (int): hidden width (all middle layers)
        n3 (int): output dimension
        depth (int): number of residual middle layers (total layers = depth + 2)
        target_ell (int): 1-indexed middle layer to study ΔW (default: 1, i.e. ℓ=2 in the paper)
        activation (str): 'relu' or 'tanh'
    """
    def __init__(self, n0=3072, d=300, n3=10, depth=1, target_ell=1, activation='relu', alpha=0, apply_Z=False):
        super().__init__()
        assert 1 <= target_ell <= depth, f"target_ell must be between 1 and depth={depth}"

        self.depth = depth
        self.target_ell = target_ell  # which middle layer to study (1-indexed)
        self.act_fn = F.relu if activation == 'relu' else torch.tanh

        self.fc_in  = nn.Linear(n0, d, bias=False)                             # W_1
        self.fc_mid = nn.ModuleList([nn.Linear(d, d, bias=False)               # W_2 ... W_{L-1}
                                     for _ in range(depth)])
        self.fc_out = nn.Linear(d, n3, bias=False)                             # W_L

        # Captured during forward for the alignment metric at target_ell
        self.h_pre  = None  # h_{target_ell}(x)        pre-activation  → numerator
        self.h_post = None  # h'_{target_ell}(x) = ϕ(h_{target_ell})  → denominator

        self.alpha=alpha
        self.apply_Z = apply_Z

        if self.apply_Z:
            self.fc_Z = nn.ModuleList([nn.Linear(d, d, bias=False)               # Z_2 ... Z_{L-1}
                                     for _ in range(depth)])
        
        self.apply(self._init_weights)

    def forward(self, x):
        h = self.fc_in(x)        # h_1(x) = W_1 x  (pre-activation)
        h_prime = self.act_fn(h) # h'_1(x)

        for i, fc in enumerate(self.fc_mid):            
            if i == self.target_ell:
                self.target_W = fc.weight.detach()
                self.h_pre = h.detach() # h^l-1
                self.phi_h_pre = h_prime.detach()
            
            if self.apply_Z:
                if i == self.target_ell:
                    self.target_Z = self.fc_Z[i].weight.detach()
                h = self.fc_Z[i](h) + (1/self.depth**self.alpha)*self.act_fn(fc(h_prime))  # residual update
            else:
                h = h + (1/self.depth**self.alpha)*self.act_fn(fc(h_prime))  # residual update
            if i == self.target_ell:
                self.h_l = h.detach() # h^l
            h_prime = self.act_fn(h)
            if i == self.target_ell:
                self.h_prime = h_prime.detach() # \phi(h^l)

        return self.fc_out(h_prime)

    def studied_layer(self, metric_for_Z):
        """Returns the weight matrix of the studied layer, for gradient access."""
        if metric_for_Z:
            return self.fc_Z[self.target_ell]
        else:
            return self.fc_mid[self.target_ell]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.01)  # faster than orthogonal
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def stable_rank(W):
    """||W||_F² / ||W||_op²"""
    W = W.float()
    frob = torch.linalg.norm(W, ord='fro').item()
    op = torch.linalg.norm(W, ord=2).item()
    return (frob / (op + 1e-12))**2
 
 
def nuclear_norm(W):
    W = W.float()
    return torch.linalg.norm(W, ord=2).item()


def alignment_sum_metric_update_res_block(delta_h, delta_W, delta_Z, h_pre_tm1, phi_h_pre_tm1, L_alpha):
    """
    ||Δ_{Z^ℓ, W^ℓ} h^{ℓ}||₂^2 / (||ΔZ^ℓ h_t^{ℓ-1}||_2^2 + ||L^-\alpha ΔW^ℓ Φ(h_t^{ℓ-1})||₂^2)
 
    Args:
        delta_h:  Δ_{Z^ℓ, W^ℓ} h^{ℓ}
        delta_W:  ΔW^ℓ
        delta_Z:  ΔZ^ℓ
        h_pre_tm1: h_t^{ℓ-1}
        phi_h_pre_tm1 : Φ(h_t^{ℓ-1})
        L_alpha = L^-\alpha
    """
    delta_h = delta_h.double()
    delta_W = delta_W.double()
    delta_Z = delta_Z.double()
    h_pre_tm1 = h_pre_tm1.double()
    phi_h_pre_tm1 = phi_h_pre_tm1.double()
    
    numer = torch.norm(delta_h, dim=1, p=2)**2

    delta_Z_h_pre_tm1 = delta_Z@h_pre_tm1.T
    delta_W_phi_h_pre_tm1 = L_alpha * delta_W @ phi_h_pre_tm1.T

    print(f"numer: {numer.mean().item():.40e} - delta_Z_h_pre_tm1: {(torch.norm(delta_Z_h_pre_tm1, dim=0, p=2)**2).mean().item():.40e} - delta_W_phi_h_pre_tm1: {(torch.norm(delta_W_phi_h_pre_tm1, dim=0, p=2)**2).mean().item():.40e}")

    denom = torch.norm(delta_Z_h_pre_tm1, dim=0, p=2)**2 + torch.norm(delta_W_phi_h_pre_tm1, dim=0, p=2)**2   # (B,)
 
    # ratio = numer / (denom)
    log_ratio = torch.log(numer) - torch.log(denom)
    ratio = torch.exp(log_ratio)

    return ratio.mean().item()



def run_trial(trainloader, activation, device, net='RESNET', depth=1, 
              target_ell=1, alpha=0, apply_Z=False, metric_for_Z=False, vector="h", 
              checkpoints=(10, 50, 100, 1000, 5000, 10000, 15000),
              unconstrained=True, scale=1.0, lr=1e-2):
    
    results = {step: {"alignment_sum_metric_update_res_block": None} for step in checkpoints}
    
    if net == 'RESNET':
        model = RESNET(activation=activation, depth=depth, target_ell=target_ell, 
                      alpha=alpha, apply_Z=apply_Z).to(device).double()
    # print(model)
    
    factor_completep_scion = (depth)**(args.alpha - 1) 
    
    # we only look at the change in the target_ell layer : fc_mid[target_ell]
    optim_groups = [{
        'params': model.fc_mid[target_ell].parameters(), 
        'norm': 'Spectral', 
        'norm_kwargs': {'steps': 5}, 
        'scale': 1.0,
        'lr_scale': factor_completep_scion
    }
    ]
    if args.apply_Z:
        factor_completep_scion_Z =  (depth) ** (- 1) 
        # same: we only lok at the change in the target_el layer
        optim_groups.append(
            {
            'params': model.fc_Z[target_ell].parameters(), 
            'norm': 'Spectral', 
            'norm_kwargs': {'steps': 5}, 
            'scale': 1.0,
            'lr_scale': factor_completep_scion_Z
            }
        )

    optimizer = Scion(optim_groups,
                      lr=args.lr, 
                      momentum=args.momentum, 
                      unconstrained=args.unconstrained
                      )

    max_steps = max(checkpoints)
    global_step = 0
    epoch = 0

    L_alpha = depth**(-alpha)

    while global_step < max_steps:
        epoch += 1
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_batches = 0

        h_pre_tm1 = None
        W_tm1 = None
        Z_tm1 = None

        for batch_num, (batch_data, batch_labels) in enumerate(trainloader):
            if global_step >= max_steps:
                break

            x = batch_data.to(device).view(batch_data.size(0), -1)
            y = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            h_before = model.h_l.detach().clone()
            h_pre_before = model.h_pre.detach().clone()
            phi_h_pre_before = model.phi_h_pre.detach().clone()

            W_before = model.studied_layer(metric_for_Z=False).weight.detach().clone()
            Z_before = model.studied_layer(metric_for_Z=True).weight.detach().clone()

            # update
            optimizer.step()
            global_step += 1

            # forward after update : step t+1
            with torch.no_grad():
                logits_after = model(x)

                h_after = model.h_l.detach().clone()
                # no h_pre after bc it is from the layer \ell -1 which is not updated here

                W_after = model.studied_layer(metric_for_Z=False).weight.detach().clone()
                Z_after = model.studied_layer(metric_for_Z=True).weight.detach().clone()
            
            # delta computation
            delta_h = h_after - h_before
            delta_W = W_after - W_before
            delta_Z = Z_after - Z_before


            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += y.size(0)
            epoch_batches += 1

            if global_step in checkpoints :
                with torch.no_grad():
                    alignment_sum_update_metric =alignment_sum_metric_update_res_block(delta_h, delta_W, delta_Z, h_pre_before, phi_h_pre_before, L_alpha)
                    results[global_step] = {
                        "alignment_sum_metric_update_res_block": alignment_sum_update_metric
                    }

                print(
                    f"> checkpoint {global_step}: alignment_sum_metric_update_res_block={alignment_sum_update_metric:.4f}"
                )

        mean_epoch_loss = epoch_loss / max(epoch_batches, 1)
        mean_epoch_acc = epoch_correct / max(epoch_total, 1)

        # print(f"Epoch {epoch:3d} | mean loss: {mean_epoch_loss:.7f} | mean acc: {mean_epoch_acc:.4f} | steps: {global_step}/{max_steps}")

    return results
        
#######################################################################################################################
## EXPERIMENT
#######################################################################################################################

import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = torchvision.datasets.CIFAR10(
    root='cifar10/', train=True, download=True, transform=transform
)

batch_size = args.batch_size
trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,      
    pin_memory=True,
    persistent_workers=True
)
print(len(dataset))

depths = list(dict.fromkeys(int(round(x)) for x in np.logspace(np.log2(4), np.log2(64), 20, base=2)))

activations = [args.act]
n_trials = args.n_trials
vector = args.vector

apply_Z = args.apply_Z
metric_for_Z = args.metric_for_Z

# checkpoints = [10, 50, 100, 1000, 5000, 10000, 15000]
checkpoints = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000]

net = args.net
target_ell=args.target_ell
alpha=args.alpha


# prepare for the saing
from collections import defaultdict
import json

results = defaultdict(
    lambda: {
        "alignment_sum_metric_update_res_block": defaultdict(list),
    }
)

# file name
json_name = "alignment_sum_metric_update_res_block"

apply_Z_str = "_apply_Z" if args.apply_Z else ""
metric_Z_str = "_metric_for_Z" if args.metric_for_Z else ""

SAVE_FILE = f"{json_name}_with_{vector}{apply_Z_str}{metric_Z_str}_lr_{args.lr}_momentum_{args.momentum}_{max(checkpoints)}_steps.json"


def save_results(results, filename):
    serializable = {
        str(step): {
            metric: {
                str(depth): values
                for depth, values in metric_dict.items()
            }
            for metric, metric_dict in step_dict.items()
        }
        for step, step_dict in results.items()
    }

    with open(filename, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"saved {filename}")

# for act in activations:
#     for metric in ['alignment_W', 'alignment_update']:
#         results[(act, metric)] = {L: [] for L in depths}

print("starting")

for trial in range(n_trials):
    indices = torch.randperm(len(dataset))[:batch_size]
    imgs, labels = zip(*[dataset[i] for i in indices])
    imgs = torch.stack(imgs).view(batch_size, -1)   # (max_B, 3072)
    labels = torch.tensor(labels)

    for act in activations:
        for depth in depths:
            print(f"\n ## Training for apply_Z {apply_Z} - metric_for_Z {metric_for_Z} - trial {trial} - act {act} - depth {depth}")
            trial_results = run_trial(trainloader, act, device, net, depth, target_ell, alpha, apply_Z, metric_for_Z, vector, checkpoints=checkpoints)
            
            # store
            for step in checkpoints:
                if trial_results[step]["alignment_sum_metric_update_res_block"] is not None:
                    results[step]["alignment_sum_metric_update_res_block"][depth].append(trial_results[step]["alignment_sum_metric_update_res_block"])
            save_results(results, SAVE_FILE)
