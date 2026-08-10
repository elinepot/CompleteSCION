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

if args.metric_for_Z and not args.apply_Z:
    assert False, f"cannot compute the metric for Z if Z is not applied" 

class MLP(nn.Module):
    """
    3-layer MLP using paper notation (ℓ is 1-indexed):
      W_1: Linear(n0, n1),  h_1(x) = W_1 x,         h'_1(x) = ϕ(h_1(x))
      W_2: Linear(n1, n2),  h_2(x) = W_2 h'_1(x),   h'_2(x) = ϕ(h_2(x))  ← ΔW here
      W_3: Linear(n2, n3),  h_3(x) = W_3 h'_2(x)    (output, no activation)
 
    For the alignment metric at ℓ=2 we need:
      h_{ℓ-1}(x)  = h_1(x)  = W_1 x              (pre-activation of layer 1)
      h'_{ℓ-1}(x) = h'_1(x) = ϕ(h_1(x))          (post-activation of layer 1)
    """
    def __init__(self, n0=3072, n1=300, n2=300, n3=10, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(n0, n1, bias=False)  # W_1
        self.fc2 = nn.Linear(n1, n2, bias=False)  # W_2  ← ΔW studied here
        self.fc3 = nn.Linear(n2, n3, bias=False)  # W_3
        self.act_fn = F.relu if activation == 'relu' else torch.tanh
 
        # Captured during forward for the alignment metric
        self.h1 = None   # h_1(x)  = W_1 x            (pre-activation, layer 1)
        # self.h1_prime = None  # h'_1(x) = ϕ(h_1(x))  (post-activation, layer 1)

        self.apply(self._init_weights)

 
    def forward(self, x):
        h1 = self.fc1(x)             # h_1(x) = W_1 x
        h1_prime = self.act_fn(h1)   # h'_1(x) = ϕ(h_1(x))
 
        self.h1 = h1.detach()              # pre-activation  → numerator of alignment
        # self.h1_prime = h1_prime.detach()  # post-activation → denominator of alignment
 
        h2 = self.fc2(h1_prime)      # h_2(x) = W_2 h'_1(x)
        h2_prime = self.act_fn(h2)   # h'_2(x) = ϕ(h_2(x))
 
        h3 = self.fc3(h2_prime)      # h_3(x) = W_3 h'_2(x)  (output)
        return h3
    
    def _init_weights(self, module):
        # either Normal or Orthogonal initialization
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.orthogonal_(module.weight)


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
            # capture the inputs to the studied layer (layer index i+1, 1-indexed = i+2 in paper)
            if i + 1 == self.target_ell:
                self.h_pre  = h.detach()       # h_{ℓ-1}(x)
                self.h_post = h_prime.detach() # h'_{ℓ-1}(x) 
            
            if i == self.target_ell:
                self.target_W = fc.weight.detach()
            
            if self.apply_Z:
                if i == self.target_ell:
                    self.target_Z = self.fc_Z[i].weight.detach()
                if i == self.target_ell:
                    self.Z_h_pre = self.fc_Z[i](h).detach()
                    self.residual_block = (1/self.depth**self.alpha)*self.act_fn(fc(h_prime)).detach()
                h = self.fc_Z[i](h) + (1/self.depth**self.alpha)*self.act_fn(fc(h_prime))  # residual update
            else:
                h = h + (1/self.depth**self.alpha)*self.act_fn(fc(h_prime))  # residual update
            if i == self.target_ell:
                self.h_lp1 = h.detach()
            h_prime = self.act_fn(h)

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


def alignment_sum_metric_res_block(h_lp1, Z_h_pre, residual_block):
    """
    ||h^{ℓ}(x)||₂^2 / (||Z^ℓ h^{ℓ-1}||_2^2 + ||L^-\alpha W^ℓ Φ(h_{ℓ-1}(x))||₂^2)
 
    Args:
        h_lp1:    h^{ℓ}
        Z_h_pre:  Z^ℓ h^{ℓ-1}
        residual_block:   L^-\alpha W^ℓ Φ(h_{ℓ-1}(x))
    """
    h_lp1 = h_lp1.float()
    Z_h_pre = Z_h_pre.float()
    residual_block = residual_block.float()


    numer = torch.norm(h_lp1, dim=1, p=2)**2

    denom = torch.norm(Z_h_pre, dim=1, p=2)**2 + torch.norm(residual_block, dim=1, p=2)**2   # (B,)
 
    ratio = numer / (denom + 1e-12)
    return ratio.mean().item()


def run_trial(trainloader, activation, device, net='RESNET', depth=1, 
              target_ell=1, alpha=0, apply_Z=False, metric_for_Z=False, vector="h", 
              checkpoints=(10, 50, 100, 1000, 5000, 10000, 15000),
              unconstrained=True, scale=1.0, lr=1e-2):
    
    results = {step: {"alignment_W": None, "alignment_update": None} for step in checkpoints}
    
    if net == 'RESNET':
        model = RESNET(activation=activation, depth=depth, target_ell=target_ell, 
                      alpha=alpha, apply_Z=apply_Z).to(device)
    else:
        model = MLP(activation=activation).to(device)
    print(model)
    
    factor_completep_scion = (depth)**(args.alpha - 1) 
    
    optim_groups = [{
    'params': model.fc_mid.parameters(), 
        'norm': 'Spectral', 
        'norm_kwargs': {'steps': 5}, 
        'scale': 1.0,
        'lr_scale': factor_completep_scion
    },{
        'params': model.fc_out.parameters(), 
        'norm': 'Sign', 
        'norm_kwargs': {}, 
        'scale': 10.0,
        'lr_scale': 1,
    },{
        'params': model.fc_in.parameters(), 
        'norm': 'Sign', 
        'norm_kwargs': {}, 
        'scale': 10.0,
        'lr_scale': 1,
    }
    ]
    if args.apply_Z:
        factor_completep_scion_Z =  (depth) ** (- 1) 
        optim_groups.append(
            {
            'params': model.fc_Z.parameters(), 
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

    while global_step < max_steps:
        epoch += 1
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_batches = 0

        for batch_num, (batch_data, batch_labels) in enumerate(trainloader):
            if global_step >= max_steps:
                break

            x = batch_data.to(device).view(batch_data.size(0), -1)
            y = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            if net == 'RESNET':
                W_layer = model.studied_layer(metric_for_Z)
                g = W_layer.weight.grad.detach()
                h_lp1 = model.h_lp1
                Z_h_pre = model.Z_h_pre 
                residual_block = model.residual_block
            else:
                g = model.fc2.weight.grad.detach()
                h_pre = model.h1
                h_post = model.h1_prime
                W = model.fc2.weight.detach()

            spectral = Spectral()
            delta_W = spectral.lmo(g.float())

            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += y.size(0)
            epoch_batches += 1

            if global_step in checkpoints :
                with torch.no_grad():
                    alignment_sum_metric =alignment_sum_metric_res_block(h_lp1, Z_h_pre, residual_block)
                    results[global_step] = {
                        "alignment_sum_metric_res_block": alignment_sum_metric
                    }

                print(
                    f"> checkpoint {global_step}: alignment_sum_metric={alignment_sum_metric:.4f}"
                )

        mean_epoch_loss = epoch_loss / max(epoch_batches, 1)
        mean_epoch_acc = epoch_correct / max(epoch_total, 1)

        print(
            f"Epoch {epoch:3d} | "
            f"mean loss: {mean_epoch_loss:.7f} | "
            f"mean acc: {mean_epoch_acc:.4f} | "
            f"steps: {global_step}/{max_steps}"
        )

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
    root='/home/ep42de8l/MLP_CIFAR10/cifar10/', train=True, download=True, transform=transform
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


depths = list(dict.fromkeys(int(round(x)) for x in np.logspace(np.log2(4), np.log2(512), 20, base=2)))

activations = [args.act]
n_trials = args.n_trials
vector = args.vector

apply_Z = args.apply_Z
metric_for_Z = args.metric_for_Z

checkpoints = [1, 2, 5, 10, 20, 50, 100,250, 500, 1000]

net = args.net
target_ell=args.target_ell
alpha=args.alpha


# prepare for the saing
from collections import defaultdict
import json

results = defaultdict(
    lambda: {
        "alignment_sum_metric_res_block": defaultdict(list),
    }
)

# file name
json_name = "alignment_sum_res_block"

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

print("starting")

for trial in range(n_trials):
    indices = torch.randperm(len(dataset))[:batch_size]
    imgs, labels = zip(*[dataset[i] for i in indices])
    imgs = torch.stack(imgs).view(batch_size, -1)   # (max_B, 3072)
    labels = torch.tensor(labels)

    for act in activations:
        for depth in depths:
            print(f"## Training for apply_Z {apply_Z} - metric_for_Z {metric_for_Z} - trial {trial} - act {act} - depth {depth}")
            trial_results = run_trial(trainloader, act, device, net, depth, target_ell, alpha, apply_Z, metric_for_Z, vector, checkpoints=checkpoints)
            
            # store
            for step in checkpoints:
                if trial_results[step]["alignment_sum_metric_res_block"] is not None:
                    results[step]["alignment_sum_metric_res_block"][depth].append(trial_results[step]["alignment_sum_metric_res_block"])
            save_results(results, SAVE_FILE)