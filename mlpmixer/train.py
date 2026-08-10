import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import wandb
import warmup_scheduler
import numpy as np

from utils import rand_bbox

from scion import  Scion, zeropower_via_newtonschulz5
from spectral_clip import jiacheng_orthogonalize
from spectral_norm_computing import compute_naive_upper_bound
from scion_utils import normalize_model

import math
import os

import sys

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.data.distributed import DistributedSampler

from datetime import datetime
from functools import partial


class Trainer(object):
    def __init__(self, model, args, writer=None, writer_norm_metrics=None, checkpoints_path=None):
        # wandb.config.update(args)
        self.device = args.device
        self.clip_grad = args.clip_grad
        self.cutmix_beta = args.cutmix_beta
        self.cutmix_prob = args.cutmix_prob
        self.model = model
        self.checkpoints_path = checkpoints_path
        print(model)
        if args.optimizer=='sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer=='adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        elif args.optimizer =='scion':
            radius_input_layer = args.radius_input_layer if args.radius_input_layer else args.L**(1/(args.depth+2))
            radius_hidden_layer = args.radiuses_hidden if args.radiuses_hidden else args.L**(1/(args.depth+2))
            radius_last_layer = args.radius_last_layer if args.radius_last_layer else args.L**(1/(args.depth+2))

            if args.model == 'mlp_mixer':
                name_first_layer = 'patch_emb.1'
                name_last_layer = 'clf'
            elif args.model == 'mlp':
                name_first_layer = 'embedding'
                name_last_layer = 'unembedding'
            
            linear_weights = []
            first_weight = []
            last_weight = []
            for name, module in model.named_modules():
                # print(name)
                if isinstance(module, nn.Linear):
                    if name == name_last_layer:
                        last_weight.append(module.weight)
                    elif name == name_first_layer:
                        first_weight.append(module.weight)
                    else:
                        linear_weights.append(module.weight)
                    # print(name, module.weight.dtype, torch.is_tensor(module.weight.data))

            print("Nb of linear weight:", len(linear_weights))
            print("Nb of last weight:", len(last_weight))
            print("Nb of first weight:", len(first_weight))

            # lr factor in function of the depth L:
            lr_factor = (2*args.num_layers)**(args.alpha - 1) 
            
            optim_groups = [{
                        'params': linear_weights,
                        'norm': 'Spectral',
                        'norm_kwargs': {'steps': 9, 'newton_function': 'jiacheng'},
                        'scale': [radius_hidden_layer]*len(linear_weights), 
                        "weight_decay":args.weight_decay,
                        "projection":args.projection,
                        "lr_scale":lr_factor,
                    },
                    {   'params': first_weight,
                        'norm': 'Spectral',
                        'norm_kwargs': {'steps': 9, 'newton_function': 'jiacheng'},
                        'scale': [radius_input_layer]*len(first_weight), 
                        "weight_decay":args.weight_decay,
                        "projection":args.projection,
                        "lr_scale":1,
                    },
                    {   'params': last_weight,
                        'norm': 'Sign',
                        'norm_kwargs': {'normalized': True},
                        'scale': [radius_last_layer]*len(last_weight), 
                        "weight_decay":args.weight_decay,
                        "projection":args.projection,
                        "lr_scale":1,
                    }
            ]

            if args.normalization == "layernorm":
                ln_weights, ln_biases = [], []

                for name, module in model.named_modules():
                    if isinstance(module, nn.LayerNorm):
                        if module.weight is not None and module.weight.requires_grad:
                            ln_weights.append(module.weight)
                        if module.bias is not None and module.bias.requires_grad:
                            ln_biases.append(module.bias)

                print("Nb of ln_weights:", len(ln_weights))
                print("Nb of ln_biases:", len(ln_biases))

                ln_biases_group = {
                        'params': ln_biases,
                        'norm': 'BiasRMS', # heuristically uses l2 for normalization layers
                        'norm_kwargs': {},
                        'scale': [radius_hidden_layer]*len(ln_biases), 
                        "weight_decay":args.weight_decay,
                        "projection":args.projection,
                        "lr_scale":1,
                    }
                
                ln_weight_group = {
                        'params': ln_weights,
                        'norm': 'Sign',
                        'norm_kwargs': {'normalized': True},
                        'scale': [radius_hidden_layer]*len(ln_weights), 
                        "weight_decay":args.weight_decay,
                        "projection":args.projection,
                        "lr_scale":1,
                    }
                
                optim_groups.append(ln_biases_group)
                optim_groups.append(ln_weight_group)

            
            print('lr factor = (2*args.num_layers)**(args.alpha - 1)  =', lr_factor)            
            self.optimizer = Scion(optim_groups, lr=args.lr, momentum=args.momentum, projection=args.projection, apply_weight_decay=args.apply_weight_decay)
            
        else:
            raise ValueError(f"No such optimizer: {self.optimizer}")

        if args.scheduler=='step':
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=args.gamma)
        elif args.scheduler=='cosine':
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        else:
            raise ValueError(f"No such scheduler: {self.scheduler}")


        if args.warmup_epoch:
            self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler
        # self.scaler = torch.amp.GradScaler('cuda')

        self.epochs = args.epochs
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), 0.

        self.writer=writer
        self.writer_norm_metrics=writer_norm_metrics
        self.args=args
    
    def _train_one_step(self, batch, gpu):
        self.model.train()
        img, label = batch
        # img = img.to(torch.bfloat16)
        self.num_steps += 1
        # img, label = img.to(self.device), label.to(self.device)
        img, label = img.cuda(gpu), label.cuda(gpu)

        self.optimizer.zero_grad()
        r = np.random.rand(1)
        if self.cutmix_beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = torch.randperm(img.size(0)).to(self.device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            # compute output
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # with torch.amp.autocast("cuda"):
                out = self.model(img)
                loss = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)
        else:
            # compute output
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # with torch.amp.autocast("cuda"):
                out = self.model(img)
                loss = self.criterion(out, label)
        # print('loss:', loss.mean())
        if torch.isnan(loss.mean()):
            print('nan in loss')
            print('y hat:', out.mean())
            torch.save(self.model.state_dict(), f'scion_{self.args.projection}_nblocks_{self.args.num_layers}lr_{self.args.lr}_min_lr_{self.args.min_lr}_momentum_{self.args.momentum}_ep_{self.args.epochs}_model_that_output_nan.pt')
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"{name} | norm of the weight: {torch.norm(param)} | mean of the weight: {param.mean()}")
                    if torch.isnan(param).any() :
                        print(f"Problem in {name}: NaN detected")
                    if torch.isinf(param).any():
                        print(f"Problem in {name}: Inf detected")
            sys.exit()
        

        # self.scaler.scale(loss).backward()
        loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        self.optimizer.step()
        self.optimizer.zero_grad()

        acc = out.argmax(dim=-1).eq(label).sum(-1)/img.size(0)
        correct = out.argmax(dim=-1).eq(label).sum()
        # wandb.log({
        #     'loss':loss,
        #     'acc':acc
        # }, step=self.num_steps)

        if self.writer:
            self.writer.add_scalar("Loss / Train", loss, self.num_steps)
            self.writer.add_scalar("Accuracy / Train", acc, self.num_steps)

        self.train_epoch_loss = loss
        self.train_epoch_acc = acc

        return loss.detach(), acc.detach(), correct.detach()



    # @torch.no_grad
    def _test_one_step(self, batch, gpu):
        self.model.eval()
        img, label = batch
        # img = img.to(torch.bfloat16)
        # img, label = img.to(self.device), label.to(self.device)
        img, label = img.cuda(gpu), label.cuda(gpu)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                out = self.model(img)
                if torch.isnan(out.mean()):
                    print('out is nan')
                    print('out: ', out.shape, out.mean(0))
                    print('img:', img.mean())
                    for name, p in self.model.named_parameters():
                        print(f'parameter {name}: mean : {p.mean()}')
                        # has_nan = torch.isnan(t).any()
                        # has_inf = torch.isinf(t).any()
                        # # print(f'{name} has nan: {has_nan} | has inf: {has_inf}')
                        # grad_norm = p.grad.norm().item() if p.grad is not None else float('nan')
                    sys.exit()
                loss = self.criterion(out, label)

        self.epoch_loss += loss * img.size(0)
        self.epoch_corr += out.argmax(dim=-1).eq(label).sum(-1)


    def fit(self, train_dl, test_dl, train_sampler, csv_logger, rank, gpu):
        self.criterion=self.criterion.cuda(gpu)
        best_epoch_loss=10000
        torch.cuda.synchronize()
        start = datetime.now()
        for epoch in range(1, self.epochs+1):
            # Train for one epoch
            train_sampler.set_epoch(epoch)

            mup_enable_coord_check_logging = True

            if mup_enable_coord_check_logging and epoch>0:
                coord_check_dict = {
                    'patch_emb': [],
                    'mlp1': [],
                    'mlp2': [],
                    'clf': [],
                    'last_layer': [],
                }
                def hook(module, input, output, key):
                    with torch.no_grad():
                        coord_check_dict[key].append(output.abs().mean().item())
            
                coord_check_handles = []
                for module_name, module in self.model.named_modules():
                    # print(f'transformer.h.{n_layer - 1}')
                    # if module_name == 'transformer.wte':
                    if module_name.endswith('patch_emb'):
                        coord_check_handles.append(module.register_forward_hook(partial(hook, key='patch_emb')))
                    elif module_name.endswith('.mlp1'):
                        coord_check_handles.append(module.register_forward_hook(partial(hook, key='mlp1')))
                    elif module_name.endswith('.mlp2'):
                        coord_check_handles.append(module.register_forward_hook(partial(hook, key='mlp2')))
                    # elif module_name == 'lm_head':
                    elif module_name.endswith('.clf'):
                        coord_check_handles.append(module.register_forward_hook(partial(hook, key='clf')))
                    # elif module_name == f'transformer.h.{n_layer - 1}':
                    elif module_name.endswith(f'mixer_layers.{self.args.num_layers - 1}'):
                        coord_check_handles.append(
                            module.register_forward_hook(partial(hook, key='last_layer'))
                        )
                    # else:
                    #     coord_check_dict = None
                    #     print("coord check dict is None")

            total_train_loss = torch.zeros(1).cuda(gpu)
            total_train_corr = torch.zeros(1).cuda(gpu)
            total_train_samples = torch.zeros(1).cuda(gpu)

            for batch in train_dl:
                loss, acc, corr = self._train_one_step(batch, gpu)

                batch_size = batch[0].size(0)

                total_train_loss += loss * batch_size
                total_train_corr += corr
                total_train_samples += batch_size
            
            # Synchronize training metrics across all GPUs
            torch.distributed.all_reduce(total_train_loss)
            torch.distributed.all_reduce(total_train_corr)
            torch.distributed.all_reduce(total_train_samples)

            train_epoch_loss = (total_train_loss / total_train_samples).item()
            train_epoch_acc = (total_train_corr / total_train_samples).item()

            self.scheduler.step()

            
            # validation performaces
            self.epoch_loss, self.epoch_corr = torch.zeros(1).cuda(gpu), torch.zeros(1).cuda(gpu)
            num_val_imgs = torch.zeros(1).cuda(gpu)

            for batch in test_dl:
                # _test_one_step modifies self.epoch_loss and self.epoch_corr
                self._test_one_step(batch, gpu)
                num_val_imgs += batch[0].size(0)

            # Synchronize validation metrics
            torch.distributed.all_reduce(self.epoch_loss)
            torch.distributed.all_reduce(self.epoch_corr)
            torch.distributed.all_reduce(num_val_imgs)

            val_epoch_loss = (self.epoch_loss / num_val_imgs).item()
            val_epoch_acc = (self.epoch_corr / num_val_imgs).item()

            
            # training performances without cutmix /!!\
            self.epoch_loss, self.epoch_corr = torch.zeros(1).cuda(gpu), torch.zeros(1).cuda(gpu)
            num_train_imgs = torch.zeros(1).cuda(gpu)

            for batch in train_dl:
                # _test_one_step modifies self.epoch_loss and self.epoch_corr
                self._test_one_step(batch, gpu)
                num_train_imgs += batch[0].size(0)

            # Synchronize validation metrics
            torch.distributed.all_reduce(self.epoch_loss)
            torch.distributed.all_reduce(self.epoch_corr)
            torch.distributed.all_reduce(num_train_imgs)

            train_no_cutmix_epoch_loss = (self.epoch_loss / num_train_imgs).item()
            train_no_cutmix_epoch_acc = (self.epoch_corr / num_train_imgs).item()


            
            # saving 
            if rank == 0:
                print(f"Epoch {epoch}/{self.epochs} | Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} | lr:  {self.optimizer.param_groups[0]['lr']}")

                checkpoint_data = {
                    'args': self.args,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss_with_cutmix': train_epoch_loss,
                    'train_loss': train_no_cutmix_epoch_loss,
                    'val_acc': val_epoch_acc
                }

                # Save Last
                torch.save(checkpoint_data, os.path.join(self.checkpoints_path, 'last_checkpoint.pt'))
                
                # Save Best
                if val_epoch_loss<= best_epoch_loss:
                    best_epoch_loss = val_epoch_loss
                    torch.save(checkpoint_data, os.path.join(self.checkpoints_path, 'best_checkpoint.pt'))

                if csv_logger:
                    log_dict = {
                        "epoch": epoch,
                        "train_with_cutmix/loss": train_epoch_loss,
                        "train_with_cutmix/acc": train_epoch_acc,
                        "train/loss": train_no_cutmix_epoch_loss,
                        "train/acc": train_no_cutmix_epoch_acc,
                        "val/loss": val_epoch_loss,
                        "val/acc": val_epoch_acc,
                        "lr": self.optimizer.param_groups[0]['lr']
                    }
                    

                    if mup_enable_coord_check_logging :
                        for key in coord_check_dict:
                            log_dict[key + '_act_abs_mean'] = np.mean(coord_check_dict[key])

                    csv_logger.log(log_dict)
                    csv_logger.step()


            if mup_enable_coord_check_logging and epoch>0:
                for handle in coord_check_handles:
                    handle.remove()


        # if rank==0:
        #     
        torch.cuda.synchronize()
        cuda_time= str(datetime.now() - start)
        print("Training complete in: " , cuda_time) 

        log_training_stats(
        file_path='cuda_time_benchmark.csv',
        gpu=self.args.gpus,
        num_workers=self.args.num_workers,
        nr=self.args.nr,
        epochs=self.args.epochs,
        time_taken=cuda_time
        )


import csv

def log_training_stats(file_path, gpu, num_workers, nr, epochs, time_taken):
    # Define the columns
    fieldnames = ['gpu', 'num_workers', 'nr', 'epochs', 'time']
    
    # Check if we need to write the header (if file doesn't exist or is empty)
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
    
    # Append mode ('a') ensures we don't delete previous runs
    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'gpu': gpu,
            'num_workers': num_workers,
            'nr': nr,
            'epochs': epochs,
            'time': time_taken
        })