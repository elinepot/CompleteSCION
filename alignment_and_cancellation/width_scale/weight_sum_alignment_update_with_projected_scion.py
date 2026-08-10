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

from spectral_clip import spectral_clip_jordan

from decimal import *
getcontext().prec = 32

@dataclass
class Hyperparameters:
    # data hyperparams
    batch_size : int = 1024
    n_trials : int = 5
    net : str = "MLP" # "RESNET", "MLP"
    act : str = 'relu' # 'relu',  'tanh'
    target_ell : int = 1
    alpha : float = 1.0
    lr : float = 1e-2
    momentum : float = 0.9
    unconstrained : bool = False
    projected : bool = True


from datargs import parse
args = parse(Hyperparameters)

print(args)

torch.set_default_dtype(torch.float64)


class MLP(nn.Module):
    def __init__(self, depth=3, n0=3072, d=300, n3=10, activation='relu', target_ell=1):
        super().__init__()
        self.fc_in = nn.Linear(n0, d, bias=False)  # W_1
        self.fc_out = nn.Linear(d, n3, bias=False)  # W_3
        self.act_fn = F.relu if activation == 'relu' else torch.tanh
 
        # Captured during forward for the alignment metric
        self.h1 = None   # h_1(x)  = W_1 x            (pre-activation, layer 1)
        # self.h1_prime = None  # h'_1(x) = ϕ(h_1(x))  (post-activation, layer 1)
        self.fc_mid = nn.ModuleList([nn.Linear(d, d, bias=False)               # W_2 ... W_{L-1}
                                     for _ in range(depth)])

        self.apply(self._init_weights)
        self.target_ell = target_ell

 
    def forward(self, x):
        h = self.fc_in(x)        # h_1(x) = W_1 x  (pre-activation)
        h_prime = self.act_fn(h) # h'_1(x)

        for i, fc in enumerate(self.fc_mid):            
            if i == self.target_ell:
                self.target_W = fc.weight.detach()
                self.h_pre = h.detach() # h^l-1
                self.phi_h_pre = h_prime.detach()
            
            h = fc(h_prime)  # residual update
            if i == self.target_ell:
                self.h_l = h.detach() # h^l
            h_prime = self.act_fn(h)
            if i == self.target_ell:
                self.h_prime = h_prime.detach() # \phi(h^l)

        return self.fc_out(h_prime)
    
    def _init_weights(self, module):
        # either Normal or Orthogonal initialization
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.orthogonal_(module.weight)
    
    def studied_layer(self):
        """Returns the weight matrix of the studied layer, for gradient access."""
        return self.fc_mid[self.target_ell]


def stable_rank(W):
    """||W||_F² / ||W||_op²"""
    W = W.float()
    frob = torch.linalg.norm(W, ord='fro').item()
    op = torch.linalg.norm(W, ord=2).item()
    return (frob / (op + 1e-12))**2
 
 
 
def matrix_norm(W):
    W = W.float()
    return torch.linalg.norm(W, ord=2).item()
 

def alignment_sum_metric_update(delta_h, delta_W, W, h_pre_tm1, phi_h_pre_tm1, delta_phi_h_pre, spectral_clipping_W_lmo, lmo_W):
    """
    ||Δ h^{ℓ}||₂^2 / (||ΔW^ℓ Φ(h_t^{ℓ-1})||_2^2 + ||W^ℓ ΔΦ(h_t^{ℓ-1})||_2^2 + ||ΔW^ℓ ΔΦ(h_t^{ℓ-1})||_2^2)

    ||(spectral_clipping(W + lr * lmo(∇W L)) - W_t^ℓ) Φ(h_t^{ℓ-1}) ||₂^2 / (||spectral_clipping(W + lr * lmo(∇W L))  Φ(h_t^{ℓ-1}) ||₂^2 + || W_t^ℓ Φ(h_t^{ℓ-1}) ||₂^2)
 
    ||W + lr * lmo(∇W L)||_S^2 / ||W||_S^2 + ||lr * lmo(∇W L)||_S^2
    
    Args:
        delta_h:  Δ_{Z^ℓ, W^ℓ} h^{ℓ}
        delta_W:  ΔW^ℓ
        h_pre_tm1: h_t^{ℓ-1}
        phi_h_pre_tm1 : Φ(h_t^{ℓ-1})
        delta_phi_h_pre : ΔΦ(h^{ℓ-1})
        spectral_clipping_W_lmo : spectral_clipping(W + lr * lmo(∇W L))
    """
    delta_h = delta_h.double()
    delta_W = delta_W.double()
    W = W.double()  
    h_pre_tm1 = h_pre_tm1.double()
    phi_h_pre_tm1 = phi_h_pre_tm1.double()
    delta_phi_h_pre = delta_phi_h_pre.double()
    spectral_clipping_W_lmo = spectral_clipping_W_lmo.double()
    
    numer = torch.norm(delta_h, dim=1, p=2)**2

    delta_W_phi_h_pre_tm1 = delta_W @ phi_h_pre_tm1.T
    W_delta_phi_h_pre = W@delta_phi_h_pre.T
    delta_W_delta_phi_h_pre = delta_W@delta_phi_h_pre.T
    
    denom = torch.norm(delta_W_phi_h_pre_tm1, dim=0, p=2)**2 + torch.norm(W_delta_phi_h_pre, dim=0, p=2)**2 +  torch.norm(delta_W_delta_phi_h_pre, dim=0, p=2)**2

    ratio = numer / denom

    # second ratio
    spectral_minus_W = spectral_clipping_W_lmo - W
    numer_spectral = torch.norm(spectral_minus_W@phi_h_pre_tm1.T, dim=0, p=2)**2

    spectral_phi = spectral_clipping_W_lmo@phi_h_pre_tm1.T
    W_phi = W@phi_h_pre_tm1.T
    denom_spectral = torch.norm(spectral_phi, dim=0, p=2)**2 + torch.norm(W_phi, dim=0, p=2)**2
    ratio_spectral = numer_spectral/denom_spectral

    # third ratio
    numer_W_lmo = matrix_norm(W+args.lr*lmo_W)**2
    denom_W_lmo = matrix_norm(W)**2 + matrix_norm(args.lr*lmo_W)**2
    ratio_W_lmo = numer_W_lmo / denom_W_lmo



    return ratio.mean().item(), ratio_spectral.mean().item(), ratio_W_lmo, numer_W_lmo



def run_trial(trainloader, activation, device, net='MLP', width=300, 
              target_ell=1, 
              checkpoints=(10, 50, 100, 1000, 5000, 10000, 15000),
              unconstrained=True, scale=1.0, lr=1e-2):
    
    results = {step: {
        "alignment_sum_update_metric": None,
        "alignment_sum_update_spectral": None,
        "alignment_sum_W_lmo": None,
        "norm_W_lmo": None} 
        for step in checkpoints}
        
    
    
    if net == 'MLP':
        model = MLP(activation=activation, d=width, target_ell=target_ell).to(device).double()
    print(model)
    
    optim_groups = [{
        'params': model.parameters(), 
        'norm': 'Spectral', 
        'norm_kwargs': {'steps': 5}, 
        'scale': 1.0,
        'lr_scale':1.0
    }
    ]
   
    optimizer = Scion(optim_groups,
                      lr=args.lr, 
                      momentum=args.momentum, 
                      unconstrained=args.unconstrained,
                      projected=args.projected
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

            # Forward before update : step t
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            h_before = model.h_l.detach().clone()
            h_pre_before = model.h_pre.detach().clone()
            phi_h_pre_before = model.phi_h_pre.detach().clone()

            W_before = model.studied_layer().weight.detach().clone()

            grad_W = model.studied_layer().weight.grad.detach().clone()
            
            spectral = Spectral()

            state_W = optimizer.state[model.studied_layer().weight].copy()
            if 'momentum_buffer' not in state_W.keys():
                print(f'momentum buffer not in state at step {global_step}')
                state_W['momentum_buffer'] = torch.zeros_like(grad_W)
            buf_W = state_W['momentum_buffer']
            buf_W.mul_(1-args.momentum).add_(grad_W, alpha=args.momentum)
            grad_W = buf_W
            lmo_W = spectral.lmo(grad_W.float())

            spectral_W_lmo = spectral_clip_jordan(W_before + lr*lmo_W, alpha=0.0, beta=1.0)

            # update
            optimizer.step()
            global_step += 1

            # forward after update : step t+1
            with torch.no_grad():
                logits_after = model(x)

                h_after = model.h_l.detach().clone()
                h_pre_before = model.h_pre.detach().clone()
                phi_h_pre_after = model.act_fn(h_pre_before).detach()
                # no h_pre after bc it is from the layer \ell -1 which is not updated here

                W_after = model.studied_layer().weight.detach().clone()
                Z_after = model.studied_layer().weight.detach().clone()
            
            # delta computation
            delta_W = W_after - W_before
            delta_h = h_after - h_before
            delta_phi_h_pre = phi_h_pre_after - phi_h_pre_before


            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += y.size(0)
            epoch_batches += 1

            if global_step in checkpoints :
                with torch.no_grad():
                    alignment_sum_update_metric, alignment_sum_update_spectral, alignment_sum_W_lmo, norm_W_lmo = alignment_sum_metric_update(delta_h, delta_W, W_before, h_pre_before, phi_h_pre_before, delta_phi_h_pre, spectral_W_lmo, lmo_W)
                    results[global_step] = {
                        "alignment_sum_update_metric": alignment_sum_update_metric,
                        "alignment_sum_update_spectral": alignment_sum_update_spectral,
                        "alignment_sum_W_lmo": alignment_sum_W_lmo,
                        "norm_W_lmo": norm_W_lmo

                    }

                print(
                    f"> checkpoint {global_step}: alignment_sum_update_metric ={alignment_sum_update_metric:.4f} - alignment_sum_update_spectral={alignment_sum_update_spectral:.4f} - alignment_sum_W_lmo={alignment_sum_W_lmo:.4f} - norm_W_lmo={norm_W_lmo:.4f}"
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
print(len(dataset))


widths = list(dict.fromkeys(int(round(x)) for x in np.logspace(np.log2(32), np.log2(2048), 20, base=2)))
activations = [args.act]
n_trials = args.n_trials

checkpoints = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000 ]

net = args.net
target_ell=args.target_ell
alpha=args.alpha


from collections import defaultdict
import json

results = defaultdict(
    lambda: {
        "alignment_sum_update_metric": defaultdict(list),
        "alignment_sum_update_spectral": defaultdict(list),
        "alignment_sum_W_lmo": defaultdict(list),
        "norm_W_lmo": defaultdict(list),
    }
)

# file name
json_name = "alignment_sum_metric_update_with_projected_scion"


SAVE_FILE = f"{json_name}_with_lr_{args.lr}_momentum_{args.momentum}_{max(checkpoints)}_steps.json"


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
        for width in widths:
            if width==2048:
                continue # avoid killed
            
            print(f"\n ## Training for trial {trial} - act {act} - width {width}")
            trial_results = run_trial(trainloader, act, device, net, width, target_ell, checkpoints=checkpoints)
            
            # store
            for step in checkpoints:
                if trial_results[step]["alignment_sum_update_metric"] is not None:
                    results[step]["alignment_sum_update_metric"][width].append(trial_results[step]["alignment_sum_update_metric"])
                if trial_results[step]["alignment_sum_update_spectral"] is not None:
                    results[step]["alignment_sum_update_spectral"][width].append(trial_results[step]["alignment_sum_update_spectral"])
                if trial_results[step]["alignment_sum_W_lmo"] is not None:
                    results[step]["alignment_sum_W_lmo"][width].append(trial_results[step]["alignment_sum_W_lmo"])
                if trial_results[step]["norm_W_lmo"] is not None:
                    results[step]["norm_W_lmo"][width].append(trial_results[step]["norm_W_lmo"])

            save_results(results, SAVE_FILE)