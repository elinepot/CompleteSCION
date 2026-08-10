import torch
from torch import nn
import torch.nn.functional as F

def spectral_norm_linear(weight):
    """Compute spectral norm (largest singular value) of a Linear layer."""
    u, s, v = torch.svd(weight.float())
    return s[0].item()


def reshape_weight_to_matrix(weight, dim=0) :
    # from https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/spectral_norm.py
    # "dimension corresponding to number of outputs, the default is 0, except for modules that are instances of ConvTranspose{1,2,3}d, when it is 1"      
    weight_mat = weight
    if dim != 0: # permute dim to front
        weight_mat = weight_mat.permute(dim, *[d for d in range(weight_mat.dim()) if d != dim])
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


def compute_weight(module, name="weight", n_power_iterations=5, eps=1e-12, dim=0, do_power_iteration=True):
    # from https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/spectral_norm.py

    assert hasattr(module, 'weight'), "Module must have a 'weight' attribute"
    weight = module.weight
    weight_mat = reshape_weight_to_matrix(weight, dim=dim)

    # Initialize u and v with a unit norm vector of appropriate size
    device = weight_mat.device
    dtype = weight_mat.dtype
    height = weight_mat.size(0)

    u = F.normalize(torch.randn(height, device=device, dtype=dtype), dim=0, eps=eps)
    v = None

    if do_power_iteration:
        with torch.no_grad():
            for _ in range(n_power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=eps, out=v)
                u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=eps, out=u)
            if n_power_iterations > 0:
                u = u.clone(memory_format=torch.contiguous_format)
                v = v.clone(memory_format=torch.contiguous_format)

    sigma = torch.dot(u, torch.mv(weight_mat, v))
  
    return sigma


def compute_spectral_norms_with_torch_PI(model, input_shape=(1, 3, 32, 32)):
    norms = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            norms[name] = spectral_norm_linear(module.weight.data)
        elif isinstance(module, nn.Conv2d):
            norms[name] = compute_weight(module, n_power_iterations=100)
    return norms


###################################################################
# SVD
###################################################################

def spectral_decomposition(W: torch.tensor):
    U, S, Vt = torch.linalg.svd(W, full_matrices=True)
    return U, S, Vt

def compute_weight_with_svd(module, dim=0):
    assert hasattr(module, 'weight'), "Module must have a 'weight' attribute"
    weight = module.weight
    weight_mat = reshape_weight_to_matrix(weight.float(), dim=dim)
    sigma = spectral_decomposition(weight_mat)[1]

    return max(sigma)

def compute_spectral_norms_with_svd(model, input_shape=(1, 3, 32, 32)):
    norms = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            d_out, d_in = module.weight.data.shape
            scale = ((d_in/d_out)**0.5)
            norms[name] = spectral_norm_linear(module.weight.data) 
        elif isinstance(module, nn.Conv2d):
            norms[name] = compute_weight_with_svd(module)
    return norms


###################################################################
# Spectral norm of convolutionnal with circular and zero padding
###################################################################

# from lip4conv : https://github.com/blaisedelattre/lip4conv/blob/main/spectral_rescaling.py#L55

def compute_spectral_rescaling_conv(kernel, n_iter=1):
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least equal to 1, got {n_iter}")
    effective_iter = 0
    kkt = kernel
    log_curr_norm = 0
    for i in range(n_iter):
        # print(f'iteration {i}/{n_iter}')
        padding = kkt.shape[-1] - 1
        # print('padding done')
        kkt_norm = kkt.norm().detach()
        kkt = kkt / kkt_norm
        # print('kkt done')
        log_curr_norm = 2 * (log_curr_norm + kkt_norm.log())
        # print('log_curr_norm done')
        # print(kkt.shape)
        kkt = kkt.half()
        # kkt = F.conv2d(kkt, kkt, padding=padding)
        conv  = torch.nn.Conv2d(in_channels=kkt.shape[0], out_channels=kkt.shape[1], kernel_size=kkt.shape[2], padding=padding, bias=False)
        conv.weight.data = kkt
        kkt = conv(kkt)
        # print('conv done')
        effective_iter += 1
    inverse_power = 2**(-effective_iter)
    t = torch.abs(kkt)
    t = t.sum(dim=(1, 2, 3)).pow(inverse_power)
    norm = torch.exp(log_curr_norm * inverse_power)
    t = t*norm
    return t

def compute_spectral_norms_with_lip4conv(model, input_shape=(1, 3, 32, 32)):
    norms = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            norms[name] = spectral_norm_linear(module.weight.data)
        elif isinstance(module, nn.Conv2d):
            rescale_weights = compute_spectral_rescaling_conv(module.weight.data, n_iter=3)
            sigma = rescale_weights.max()
            # print(name, sigma)
            norms[name] = sigma
    return norms


###################################################################
# Computing different norms
###################################################################

def compute_naive_upper_bound(model, svd=True, all_norms=False, verbose=False):
    # print('\nwith SVD Decomposition : ')
    norms = compute_spectral_norms_with_svd(model)
    p=1
    for name, norm in norms.items():
        if verbose:
            print(f"Spectral norm of {name}: {norm:.4f}")
        p*=norm
    # print('product of the norms : ', p)
    if all_norms:
        return p, norms
    return p


###################################################################
# Check is a matrix is conv
###################################################################
import numpy as np

def is_toeplitz(mat, tol=1e-5):
    """Check if a matrix is Toeplitz ie constant diagonals"""
    rows, cols = mat.shape
    for i in range(1, rows):
        for j in range(1, cols):
            if abs(mat[i, j] - mat[i-1, j-1]) > tol:
                return False
    return True

def is_bttb(matrix, block_size, tol=1e-5):
    """ Check if a matrix is Block Toeplitz with Toeplitz Blocks"""
    bh, bw = block_size
    H, W = matrix.shape
    nb_h = H // bh
    nb_w = W // bw

    if H % bh != 0 or W % bw != 0:
        print("Matrix size not divisible by block size.")
        return False

    # Extract all blocks
    blocks = [[matrix[i*bh:(i+1)*bh, j*bw:(j+1)*bw] for j in range(nb_w)] for i in range(nb_h)]

    # Check each block is Toeplitz
    for row in blocks:
        for block in row:
            if not is_toeplitz(block, tol=tol):
                print('not toeplitz block ;', block)
                return False

    print('ok here')
    # Check block rows form Toeplitz structure
    print('blocks ;', nb_h, nb_w)

    for i in range(1, nb_h):
        for j in range(1, nb_w):
            if not np.allclose(blocks[i][j], blocks[i-1][j-1], atol=1e-4):
                print('not toeplitz block ;', 'i :', i , 'j :', j)
                print('blocks[i][j]',blocks[i][j])
                print('blocks[i-1][j-1]', blocks[i-1][j-1])
                return False

    return True

import scipy.linalg as linalg 

def toeplitz_1_ch(kernel, input_size):
    """ from https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n"""
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = i_h-k_h+1, i_w-k_w+1

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz = []
    for r in range(k_h):
        toeplitz.append(linalg.toeplitz(c=(kernel[r,0], *np.zeros(i_w-k_w)), r=(*kernel[r], *np.zeros(i_w-k_w))) ) 

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(toeplitz):
        for j in range(o_h):
            W_conv[j, :, i+j, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv

def toeplitz_mult_ch(kernel, input_size):
    """
    from https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
    Compute toeplitz matrix for 2d conv with multiple in and out channels.
    Args:
        kernel: shape=(n_out, n_in, H_k, W_k)
        input_size: (n_in, H_i, W_i)"""

    kernel_size = kernel.shape
    output_size = (kernel_size[0], input_size[1] - (kernel_size[1]-1), input_size[2] - (kernel_size[2]-1))
    T = np.zeros((output_size[0], int(np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))

    for i,ks in enumerate(kernel):  # loop over output channel
        for j,k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_1_ch(k, input_size[1:])
            T[i, :, j, :] = T_k

    T.shape = (np.prod(output_size), np.prod(input_size))

    return T




if __name__ == '__main__':
    # conv = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2)

    # weight = conv.weight.data
    # # weight = weight.numpy()

    # print('conv shape : ', weight.shape)

    # T = toeplitz_mult_ch(weight, (2,3,3))
    # print('Toeplitz matrix shape : ', T.shape)

    
    # print('Is BBT : ', is_bttb(T, (3,3)))

    inp = torch.randn(1, 3, 4, 4)
    # print('inp :', inp)
    w = torch.randn(2, 3, 2, 2)
    # print('w : ', w)
    inp_unf = torch.nn.functional.unfold(inp, (2, 2)) # input, kernel size
    print('inp_unf : ', inp_unf.shape)
    M = w.view(w.size(0), -1).t()
    print('M :', M)
    print('M shape :', M.shape)
    out_conv = torch.nn.functional.conv2d(inp, w)
    print('out_conv : ', out_conv.shape)
    out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    out = out_unf.view(1, 2, 3, 3)
    print('out : ', out.shape)
    print('diff : ', (out_conv - out).abs().max())
