import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR only

import argparse
import torch

from dataloader import get_dataloaders
from utils import get_model
from train import Trainer
from spectral_norm_computing import compute_naive_upper_bound

from torch.utils.tensorboard import SummaryWriter

import time

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader

import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='c10', choices=['c10', 'c100', 'svhn'])
parser.add_argument('--model', default='mlp_mixer', choices=['mlp_mixer', 'mlp'])
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--eval-batch-size', type=int, default=1024)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--epochs', type=int, default=300)
# parser.add_argument('--precision', type=int, default=16)

## MLP MIXER PARAMETERS
parser.add_argument('--patch-size', type=int, default=4)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--hidden-c', type=int, default=512) # /!\ hidden_c = 512 and hidden_s=64 -> 1.1M parameters  # hidden size in MLP 2 of MixerLayer
parser.add_argument('--hidden-s', type=int, default=64) # hidden size in MLP 2 of MixerLayer
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--drop-p', type=int, default=0.) # dropout layers, 0 in the original code & paper
parser.add_argument('--off-act', action='store_true', help='Disable activation function')
parser.add_argument('--is-cls-token', action='store_true', help='Introduce a class token.')

## MLP PARAMETERS
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--width', type=int, default=64) 


parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--min_lr', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=0.3)
parser.add_argument('--optimizer', default='scion', choices=['adam', 'sgd', 'scion'])
parser.add_argument('--scheduler', default='cosine', choices=['step', 'cosine'])
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--off-nesterov', action='store_true')
parser.add_argument('--label-smoothing', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--warmup-epoch', type=int, default=5)
parser.add_argument('--autoaugment', action='store_true', default=True) 
parser.add_argument('--clip-grad', type=float, default=0, help="0 means disabling clip-grad")
parser.add_argument('--cutmix-beta', type=float, default=1.0)
parser.add_argument('--cutmix-prob', type=float, default=0.5)

parser.add_argument('--projection', type=str, default=None)
parser.add_argument('--radiuses_hidden', type=float, default=None)
parser.add_argument('--radius_input_layer', type=float, default=None)
parser.add_argument('--radius_last_layer', type=float, default=None)
parser.add_argument('--L', type=float, default=None)

parser.add_argument('--apply_weight_decay', action='store_true', help="Enable weight decay")
parser.add_argument('--no_apply_weight_decay', dest='apply_weight_decay',
                    action='store_false', help="Disable weight decay")
parser.set_defaults(apply_weight_decay=False)

parser.add_argument('--ortho_init', action='store_true', help="Semi orthogonal initialisation")
parser.add_argument('--no_ortho_init', dest='ortho_init',
                    action='store_false', help="No Semi orthogonal initialisation")
parser.set_defaults(ortho_init=True)

parser.add_argument('--normalization', type=str, default="rmsnorm", choices=["rmsnorm", "layernorm", "batchnorm", "none"])

parser.add_argument('--convex_skip_connection', action='store_true', help="Enable skip connection")
parser.add_argument('--no_convex_skip_connection', dest='convex_skip_connection',
                    action='store_false', help="Disable skip connection")
parser.set_defaults(convex_skip_connection=True)

parser.add_argument('--sqrt_convex_skip_connection', action='store_true', help="Enable skip connection with 1/sqrt(L)")
parser.add_argument('--no_sqrt_convex_skip_connection', dest='sqrt_convex_skip_connection',
                    action='store_false', help="Disable sqrt skip connection")
parser.set_defaults(sqrt_convex_skip_connection=False)

parser.add_argument('--root', default='pipo', type=str)
parser.add_argument('--cp-path', default='checkpoints', type=str)
parser.add_argument('--log-path', default='../logs/training_log.txt', type=str)
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)

parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='Number of node')
parser.add_argument('-g', '--gpus', default=4, type=int,
                    help='Number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='Ranking within the nodes')
parser.add_argument('-c', '--cpus', default=0, type=int,
                    help='Number of cpus per node')
parser.add_argument('-ip', '--ip', default="0.0.0.0", type=str,
                        help='Master IP address')


args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.nesterov = not args.off_nesterov

torch.random.manual_seed(args.seed)

if args.L is None and args.radiuses_hidden is None:
    raise ValueError(f"Pas de rayons selectionnés")



def loss_acc(model, dataloader, criterion, device):
    model.eval()
       
    num_imgs = 0.
    loss, corr, acc = 0., 0., 0.
    for batch in dataloader:
        img, label = batch
        img, label = img.to(device), label.to(device)

        with torch.no_grad():
            out = model(img)
            loss = criterion(out, label)

        loss += loss * img.size(0)
        corr += out.argmax(dim=-1).eq(label).sum(-1)
        
        num_imgs += batch[0].size(0)
    
    loss /= num_imgs
    acc = corr / num_imgs


    return loss, acc

def run(gpu, args):
    # Configure
    args.world_size = args.gpus * args.nodes  # Total number of processes to run
    os.environ['MASTER_ADDR'] = args.ip  # '10.89.87.67' # Master IP address
    os.environ['MASTER_PORT'] = '8888'  # Master port

    # Needed for multi-GPU
    torch.manual_seed(args.seed)


    SAVE = True

    # Process initialization
    rank = args.nr * args.gpus + gpu  # Compute rank of the process
    # rank = int(os.environ.get("SLURM_NODEID")) * args.gpus + gpu  # Compute rank of the process
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    if SAVE :
        timestr = time.strftime("%Y%m%d-%H%M%S")

        weight_decay_info = f"_weight_decay_{args.weight_decay}" if args.apply_weight_decay else ''
        model_info = f"_mlpmixer_nblocks_{args.num_layers}" if args.model=='mlp_mixer' else f"_mlp_depth_{args.depth}_width_{args.width}"
        constraints_info = f"_L_{args.L}" if args.L else f"_in_radius_{args.radius_input_layer}_hidden_radiuses_{args.radiuses_hidden}_out_radius_{args.radius_last_layer}"
        optimizer_info = 'unconstrained_scion' if (args.optimizer=='scion' and args.projection == None) else f'{args.optimizer}_{args.projection}'
        alpha=0.5 if args.sqrt_convex_skip_connection else 1.0
        args.alpha = alpha
        print(f'alpha: {alpha} | args.sqrt_convex_skip_connection: {args.sqrt_convex_skip_connection}')
        
        checkpoint_name = optimizer_info + model_info + f'_alpha_{alpha}_lr_{args.lr}_momentum_{args.momentum}_min_lr_{args.min_lr}' + constraints_info + weight_decay_info + f'_ep_{args.epochs}_seed_{args.seed}' 
                        # + timestr
        
        checkpoints_path = os.path.join(args.root, args.cp_path, checkpoint_name)
        os.makedirs(checkpoints_path, exist_ok=True)
        checkpoint_save_path = os.path.join(checkpoints_path, '_checkpoint.pt')
        # print('folder created')

        
        log_dir = os.path.join(args.root + "/runs", timestr + checkpoint_name)
        # writer = SummaryWriter(log_dir=log_dir)
        # writer_norm_metrics = SummaryWriter(log_dir=os.path.join(args.root + "/runs_norms_FW", timestr + checkpoint_name))
        writer=None
        writer_norm_metrics=None

    else:
        checkpoints_path = None
        writer = None
        writer_norm_metrics = None

    # ////////////////////////////////////////////////
    # Device configuration
    torch.cuda.set_device(gpu)

    train_dl, test_dl, train_dataset, test_dataset = get_dataloaders(args)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size * 100, shuffle=False)

    model = get_model(args)
    
    print('model loaded') # the model is automatically semi orthogonal initilaized
    
    model = model.to(torch.bfloat16)
    print('instanciating DDP')
    model = DDP(model, device_ids=[gpu])
    print('DDP done')

    # print('naive upper bound: ', compute_naive_upper_bound(model, all_norms=False))

    csv_log=True
    if rank==0 and csv_log:
        from csv_logging import CSVLogWrapper
        def log(log_dict):
            pass
        csv_logger = CSVLogWrapper(log, config=vars(args), out_dir=checkpoints_path)
    else:
        csv_logger=None

    print('\n ####### Start training ###### \n')
    trainer = Trainer(model, args, writer, writer_norm_metrics, checkpoints_path)
    trainer.fit(train_dl, test_dl, train_sampler, csv_logger, rank, gpu)


if __name__=='__main__':
    t1=time.time()
    print('args', args)
    mp.spawn(run, nprocs=args.gpus, args=(args,))
    print('time: ', time.time()-t1)
    