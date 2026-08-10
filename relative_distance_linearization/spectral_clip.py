'''
from https://leloykun.github.io/ponder/spectral-clipping/ .

We build a function that clips the spectral values of a matrix W.
Let us define S =   [0, W, 
                    Wt, 0]
and [P*, Q*
    Q*t, R*]  = orthogonalize_via_Newton_Schulz(I + S)

Then : spectral_clip_[-1,1](W) = Q* + W R*
'''

import torch

import time

# from https://kellerjordan.github.io/posts/muon/
def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # X = G
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def spectral_decomposition(W: torch.tensor):
    U, S, Vt = torch.linalg.svd(W, full_matrices=True)
    return U, S, Vt

####################################################################################
## 1. NAIVE
####################################################################################

def naive_spectral_clip(W: torch.tensor, sigma_min: float=-1., sigma_max: float=6):
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    S_clipped = torch.clip(S, min=sigma_min, max=sigma_max)
    return U @ torch.diag(S_clipped) @ Vt


####################################################################################
## 2. SPECTRAL CLIP WITH NEWTON SCHULZ
####################################################################################

def spectral_clip_jordan(W: torch.tensor, alpha: float=-1., beta: float=6.):
    # print('Sigma W_spectral_clip_NS_jiacheng :', spectral_decomposition(W)[1].max())
    I_m = torch.eye(W.shape[0])

    reshape=False
    if len(W.shape) > 2:
        reshape=True
        orig_shape = W.shape
        W = W.reshape(len(W), -1)

    msign_W = newtonschulz5(W, steps=5).to(W.dtype)

    a = (alpha + beta)*I_m.to(W.dtype).to(W.device)
    b = (alpha*msign_W - W) @ (newtonschulz5(alpha * msign_W - W).to(W.dtype)).T 
    c = (beta *msign_W - W) @ (newtonschulz5(beta  * msign_W - W).to(W.dtype)).T 

    Q = (1/2) * (a + b -c) @ msign_W

    if reshape :
        Q = Q.view(orig_shape)

    # print('Sigma Q_spectral_clip_NS_jiacheng :', spectral_decomposition(Q)[1].max())
    
    return Q

####################################################################################
## 4. HIGHAM ANTI BLOCK DIAGONAL MATRIX
####################################################################################

def spectral_clip_for_higham_jordan(W):
    W = W.to(torch.bfloat16)
    m, n = W.shape
    I = torch.eye(m + n)
    S = torch.cat((torch.cat((torch.zeros((m, m)), W), dim=1), 
                   torch.cat((W.T, torch.zeros((n, n))), dim=1)))
    gS = (1/2) * (
        (I + S) @ newtonschulz5 (I + S).float()
        - (I - S) @ newtonschulz5 (I - S).float()
    )
    return gS[:m, m:]  # read off the top-right block

def higham_spectral_clip_jordan(W, sigma_max =6.):
    return sigma_max * spectral_clip_for_higham_jordan(W / sigma_max)


####################################################################################
## 4. ABSTRACT ALGEBRA AND SYMMETRY
####################################################################################

def block_matmul(
    P1: torch.tensor, Q1: torch.tensor, R1: torch.tensor,
    P2: torch.tensor, Q2: torch.tensor, R2: torch.tensor,
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    P = P1 @ P2   + Q1 @ Q2.T
    Q = P1 @ Q2   + Q1 @ R2
    R = Q1.T @ Q2 + R1 @ R2
    return P, Q, R

def newton_schulz_iter_block(
    P: torch.tensor, Q: torch.tensor, R: torch.tensor,
    a: float, b: float, c: float,
):
    I_P = a * torch.eye(P.shape[0], dtype=P.dtype)
    I_R = a * torch.eye(R.shape[0], dtype=R.dtype)
    P2, Q2, R2 = block_matmul(P, Q, R, P, Q, R)
    P4, Q4, R4 = block_matmul(P2, Q2, R2, P2, Q2, R2)
    Ppoly = I_P + b * P2 + c * P4
    Qpoly =       b * Q2 + c * Q4
    Rpoly = I_R + b * R2 + c * R4
    return block_matmul(P, Q, R, Ppoly, Qpoly, Rpoly)

def _algebra_spectral_clip(W): # clip entre [-1, 1]
    a, b, c = (3.4445, -4.7750, 2.0315)
    P, Q, R = torch.eye(W.shape[0]), W, torch.eye(W.shape[0])
    P, Q, R = newton_schulz_iter_block(P, Q, R, a, b, c)
    return Q + W @ R

def algebra_spectral_clip(W, sigma_max=6.):
    return sigma_max * _algebra_spectral_clip(W/sigma_max)



####################################################################################
## -- JIACHENG 6 STEPS
####################################################################################

def jiacheng_orthogonalize(M, steps=6):
   # by @YouJiacheng (with stability loss idea from @leloykun)
   # https://twitter.com/YouJiacheng/status/1893704552689303901
   # https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae

    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]
    
    if len(M.shape) > 2:
        M = M.reshape(len(M), -1)
    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / torch.linalg.norm(M)
    for a, b, c in abc_list:
        A = M.T @ M
        I = torch.eye(A.shape[0], device=M.device)
        temp = (a * I + b * A + c * A @ A).to(M.dtype)
        M = M @ temp
    if transpose:
        M = M.T
    return M

def matmul_chunked(A, B, chunk_size):
    N, K = A.shape
    K2, M = B.shape
    assert K == K2

    C = torch.empty((N, M), device=A.device, dtype=A.dtype)

    for j in range(0, M, chunk_size):
        j_end = min(j + chunk_size, M)
        C[:, j:j_end] = A @ B[:, j:j_end]

    return C


# 2. SPECTRAL CLIP 
def spectral_clip_jiacheng(W: torch.tensor, alpha: float=-1., beta: float=6.):
    reshape=False
    if len(W.shape) > 2:
        reshape=True
        orig_shape = W.shape
        W = W.reshape(len(W), -1)
    I_m = torch.eye(W.shape[0])

    msign_W = jiacheng_orthogonalize(W)

    a = (alpha + beta)*I_m.to(W.dtype).to(W.device)
    b = (alpha*msign_W - W) @ (jiacheng_orthogonalize(alpha * msign_W - W)).T 
    c = (beta *msign_W - W) @ (jiacheng_orthogonalize(beta  * msign_W - W)).T 

    # we need to compute Q = (1/2) * (a + b -c) @ msign_W
    # where a = (alpha + beta)*I_m.to(W.dtype).to(W.device)
    # with I_m = torch.eye(W.shape[0]).
    # -> problem when W is very large.
    # instead: 
    # b = b.clone()
    # print('b clones')
    # b.diagonal().add_(alpha+beta)
    # print('a added')

    Q = (1/2) * (a + b -c) @ msign_W

    if reshape :
        Q = Q.view(orig_shape)

    return Q




# HIGHAM
def _spectral_clip_jiacheng(W):
    W = W.to(torch.bfloat16)
    m, n = W.shape
    I = torch.eye(m + n)
    S = torch.cat((torch.cat((torch.zeros((m, m)), W), dim=1), 
                   torch.cat((W.T, torch.zeros((n, n))), dim=1)))
    gS = (1/2) * (
        (I + S) @ jiacheng_orthogonalize (I + S).float()
        - (I - S) @ jiacheng_orthogonalize (I - S).float()
    )
    return gS[:m, m:]  # read off the top-right block

def higham_spectral_clip_jiacheng(W, sigma_max =6.):
    return sigma_max * _spectral_clip_jiacheng(W / sigma_max)


####################################################################################
## -- KEXUE & JIACHENG 6 STEPS
## from https://kexue.fm/archives/11059
####################################################################################
def msign(x, steps=10, eps=1e-20):
    """The coefficients come from https://kexue.fm/archives/10996
    """
    x = x/ torch.linalg.norm(x)
    abc = [
        (8.287212018145622, -23.59588651909882, 17.300387312530923),
        (4.107059111542197, -2.9478499167379084, 0.54484310829266),
        (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
        (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
        (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
        (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
        (1.875, -1.25, 0.375)
    ]
    y = x.mT if x.shape[-2] > x.shape[-1] else x
    y = y * torch.sqrt((y**2).sum(axis=[-2, -1], keepdims=True) + eps)
    for a, b, c in abc[:steps] + max(steps - 7, 0) * abc[-1:]:
        a, b, c = a / 1.01, b / 1.01**3, c / 1.01**5
        y = a * y + (b * (u := y @ y.mT) + c * u @ u) @ y
    return y.mT if x.shape[-2] > x.shape[-1] else y


def spectral_clip_kexue_jianlin(W: torch.tensor, beta: float=6.):
    reshape=False
    if len(W.shape) > 2:
        reshape=True
        orig_shape = W.shape
        W = W.reshape(len(W), -1)

    I_n = torch.eye(W.shape[-1]).to(W.dtype).to(W.device)

    msign_W = msign(W)
    msign_WTW_betaI = msign(W.T@W - beta*I_n)

    a = beta*msign_W + W
    b = (beta*msign_W - W)@msign_WTW_betaI

    Q = (1/2) * (a + b)

    if reshape :
        Q = Q.view(orig_shape)

    return Q


def spectral_clip_jianlin(W: torch.tensor, beta: float=6.):
    reshape=False
    if len(W.shape) > 2:
        reshape=True
        orig_shape = W.shape
        W = W.reshape(len(W), -1)

    I_n = torch.eye(W.shape[-1]).to(W.dtype).to(W.device)

    msign_W = jiacheng_orthogonalize(W)
    msign_WTW_betaI = jiacheng_orthogonalize(W.T@W - beta*I_n)

    a = beta*msign_W + W
    b = (beta*msign_W - W)@msign_WTW_betaI

    Q = (1/2) * (a + b)

    if reshape :
        Q = Q.view(orig_shape)

    return Q

def mclip3_jianlin(m, beta):
    """3rd version (3 non-nested msign)
    """
    I_n = torch.eye(m.shape[-1]).to(m.dtype).to(m.device)
    m= m/beta

    ms1 = msign(m)
    ms2 = msign(m.mT @ m + I_n)
    ms3 = msign(m.mT @ m - I_n)
    unit_clipping_W_over_beta = ((ms1 + m) @ ms2  + (ms1 - m) @ ms3) / 2 # eq (12)

    return beta*unit_clipping_W_over_beta


def mclip3_jiacheng(m, beta):
    """3rd version (3 non-nested msign)
    """
    I_n = torch.eye(m.shape[-1]).to(m.dtype).to(m.device)
    m= m/beta

    ms1 = jiacheng_orthogonalize(m)
    ms2 = jiacheng_orthogonalize(m.mT @ m + I_n)
    ms3 = jiacheng_orthogonalize(m.mT @ m - I_n)
    unit_clipping_W_over_beta = ((ms1 + m) @ ms2  + (ms1 - m) @ ms3) / 2 # eq (12)

    return beta*unit_clipping_W_over_beta



def spectral_clip_jianlin(W: torch.tensor, alpha: float=-1., beta: float=6.):
    reshape=False
    if len(W.shape) > 2:
        reshape=True
        orig_shape = W.shape
        W = W.reshape(len(W), -1)

    I_m = torch.eye(W.shape[0])

    msign_W = msign(W)

    a = (alpha + beta)*I_m.to(W.dtype).to(W.device)
    b = (alpha*msign_W - W) @ (msign(alpha * msign_W - W)).T 
    c = (beta *msign_W - W) @ (msign(beta  * msign_W - W)).T 

    Q = (1/2) * (a + b -c) @ msign_W

    if reshape :
        Q = Q.view(orig_shape)

    return Q

####################################################################################
## -- SOFT CAP
####################################################################################
# from https://github.com/Arongil/lipschitz-transformers/blob/main/nanogpt/train_spectral_cap.py

def soft_cap(M, alpha: float) :
    """Apply min(1, x) approximately to the singular values of a single matrix."""
    # Handle batched matrices by flattening batch dimensions
    orig_shape = M.shape
    if M.ndim > 2:
        M = M.reshape(-1, M.shape[-2], M.shape[-1])
    orig_dtype = M.dtype
    M = M.bfloat16()
    coeffs = [
        (1, -alpha),
        (1, alpha),
    ]
    transpose = M.shape[-1] > M.shape[-2]
    if transpose:
        M = M.mT
    for a, b in coeffs:
        A = M @ M.mT
        M = a * M + b * A @ M
    if transpose:
        M = M.mT
    
    if len(orig_shape) > 2:
        M = M.reshape(orig_shape)
    return M.to(orig_dtype)

def soft_cap_coupling(w_max: float, wd: float, max_update_norm: float) -> float:
    """Calculates the strength for soft cap that bounds singular values at w_max."""
    k = w_max * (1 - wd) + max_update_norm
    coeffs = torch.tensor([-k**9, 3 * k**7, -3 * k**5, 0.0, k - w_max], dtype=torch.float32)
    monic_coeffs = coeffs / coeffs[0]
    n = monic_coeffs.numel() - 1
    comp = torch.zeros((n, n), dtype=torch.float32)
    comp[1:, :-1] = torch.eye(n - 1)
    comp[0, :] = -monic_coeffs[1:]
    roots = torch.linalg.eigvals(comp)
    is_real = torch.abs(roots.imag) < 1e-6
    is_nonnegative = roots.real >= 0
    padded_reals = torch.where(is_real & is_nonnegative, roots.real, torch.ones_like(roots.real))
    return float(torch.min(padded_reals))

################################################################################

if __name__ == "__main__":
    W = torch.tensor([[9.0, 0.0, 1.0],
                    [-1.5, 6.0, 0.0],
                    [2.0, 13.0, -5.0],
                    [2.0, 1.0, 3.0]])

    W = torch.rand([50000, 700]).T
    
    # print('W : ', W, '\n', W.shape)
    # U, S, Vt = spectral_decomposition(W)
    # print('\nS :', S)
    # # print('U @ Vt :', U @ Vt)
    # # print('NS(W) :', newtonschulz5(W, 5))
    # # print('Jiacheng(W) :', jiacheng_orthogonalize(W))
    sigma_max = 6
    print(sigma_max)
    # print('\n-> We want sigma_max = ', sigma_max)

    # print('\n', '-'*50, '\n')
    # time_1 = time.time()
    # W_clipped_naive = naive_spectral_clip(W, sigma_max=sigma_max)
    # time_2 = time.time()
    # duration = time_2 - time_1
    # print('W_clipped_naive : ', W_clipped_naive)
    # print('Sigma W_clipped_naive :', spectral_decomposition(W_clipped_naive)[1])
    # print('Duration : ', duration)
    # print('\n', '-'*50, '\n')

    # # time_1 = time.time()
    # # W_spectral_clip_NS = spectral_clip_jordan(W, beta=sigma_max)
    # # time_2 = time.time()
    # # duration = time_2 - time_1
    # # print('W_spectral_clip_NS : ', W_spectral_clip_NS)
    # # print('Sigma W_spectral_clip_NS :', spectral_decomposition(W_spectral_clip_NS)[1])
    # # print('Duration : ', duration)
    # # print('\n', '-'*50, '\n')

    # print('### Original Spectral Clip ###')


    time_1 = time.time()
    W_spectral_clip_NS_jiacheng = spectral_clip_jiacheng(W, beta=sigma_max)
    time_2 = time.time()
    duration = time_2 - time_1
    print('W_spectral_clip_NS_jiacheng : ', W_spectral_clip_NS_jiacheng)
    print('Sigma W_spectral_clip_NS_jiacheng :', spectral_decomposition(W_spectral_clip_NS_jiacheng)[1])
    print('Duration : ', duration)
    print('\n', '-'*50, '\n')

    
    time_1 = time.time()
    W_spectral_clip_jianlin = spectral_clip_jianlin(W, beta=sigma_max)
    time_2 = time.time()
    duration = time_2 - time_1
    print('W_spectral_clip_jianlin : ', W_spectral_clip_jianlin)
    print('Sigma W_spectral_clip_jianlin :', spectral_decomposition(W_spectral_clip_jianlin)[1])
    print('Duration : ', duration)
    print('\n', '-'*50, '\n')
    

    print('### Jianlin Spectral Clip 1 ###')

    time_1 = time.time()
    W_spectral_clip_kexue_jiacheng = spectral_clip_jianlin(W, beta=sigma_max)
    time_2 = time.time()
    duration = time_2 - time_1
    print('W_spectral_clip_kexue_jiacheng : ', W_spectral_clip_kexue_jiacheng)
    print('Sigma W_spectral_clip_kexue_jiacheng :', spectral_decomposition(W_spectral_clip_kexue_jiacheng)[1])
    print('Duration : ', duration)
    print('\n', '-'*50, '\n')

    
    time_1 = time.time()
    W_sspectral_clip_kexue_jianlin = spectral_clip_kexue_jianlin(W, beta=sigma_max)
    time_2 = time.time()
    duration = time_2 - time_1
    print('W_sspectral_clip_kexue_jianlin : ', W_sspectral_clip_kexue_jianlin)
    print('Sigma W_sspectral_clip_kexue_jianlin :', spectral_decomposition(W_sspectral_clip_kexue_jianlin)[1])
    print('Duration : ', duration)
    print('\n', '-'*50, '\n')


    print('### Jianlin Spectral Clip 2 ###')

    time_1 = time.time()
    W_spectral_clip_mclip3_jianlin = mclip3_jianlin(W, beta=sigma_max)
    time_2 = time.time()
    duration = time_2 - time_1
    print('W_spectral_clip_mclip3_jianlin : ', W_spectral_clip_mclip3_jianlin)
    print('Sigma W_spectral_clip_mclip3_jianlin :', spectral_decomposition(W_spectral_clip_mclip3_jianlin)[1])
    print('Duration : ', duration)
    print('\n', '-'*50, '\n')

    
    time_1 = time.time()
    W_spectral_clip_mclip3_jiacheng = mclip3_jiacheng(W, beta=sigma_max)
    time_2 = time.time()
    duration = time_2 - time_1
    print('W_spectral_clip_mclip3_jiacheng : ', W_spectral_clip_mclip3_jiacheng)
    print('Sigma W_spectral_clip_mclip3_jiacheng :', spectral_decomposition(W_spectral_clip_mclip3_jiacheng)[1])
    print('Duration : ', duration)
    print('\n', '-'*50, '\n')
    
    # O = newtonschulz5(W).float()
    # print('O', O)
    # print('Spectral Decomposition :')
    # Sigma_O = spectral_decomposition(O)
    # print('O', torch.sign(Sigma_O))

    # time_1 = time.time()
    # alpha=soft_cap_coupling(w_max=sigma_max, wd=0, max_update_norm=100)
    # W_spectral_clip_soft_cap = soft_cap(W, alpha=alpha)
    # time_2 = time.time()
    # duration = time_2 - time_1
    # print('W_spectral_clip_soft_cap : ', W_spectral_clip_soft_cap)
    # print('Sigma W_spectral_clip_soft_cap :', spectral_decomposition(W_spectral_clip_soft_cap)[1])
    # print('Duration : ', duration)
    # print('\n', '-'*50, '\n')