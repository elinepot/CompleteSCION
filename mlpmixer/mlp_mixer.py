import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchsummary

import sys
import math
from torch.utils.checkpoint import checkpoint

class MLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, 
                 hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, 
                 num_classes=10, drop_p=0., off_act=False, is_cls_token=False,
                 normalization="rmsnorm", embedding="linear",
                 convex_skip_connection=True,
                 sqrt_convex_skip_connection=False):
        super(MLPMixer, self).__init__()

        self.hidden_size=hidden_size

        norm_dict = {
            "layernorm": lambda hidden_size: nn.LayerNorm(hidden_size),
            "batchnorm": lambda hidden_size: nn.BatchNorm1d(hidden_size),
            "none": lambda hidden_size: nn.Identity(),
            "rmsnorm": lambda hidden_size: nn.RMSNorm(hidden_size, elementwise_affine=False)
        }

        self.normalization = normalization
        self.ln = norm_dict[normalization](hidden_size)

        self.convex_skip_connection = convex_skip_connection
        self.sqrt_convex_skip_connection = sqrt_convex_skip_connection
        # print('MLP MIXER: ', self.sqrt_convex_skip_connection)

        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token

        
        if embedding=="convolution":  # CONVOLUTION EMBEDDING
            self.patch_emb = nn.Sequential(
                nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
                Rearrange('b d h w -> b (h w) d')
            )

        else:  # LINEAR EMBEDDING
            self.patch_emb = nn.Sequential(
                Rearrange('b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch_size, pw=patch_size),  #turn the image in a batch of flattened patches, of shape (c * p * p)
                nn.Linear(in_channels * patch_size * patch_size, hidden_size)
            )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1


        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act, normalization, convex_skip_connection, sqrt_convex_skip_connection, num_layers) 
            for _ in range(num_layers)
            ]
        )

        self.clf = nn.Linear(hidden_size, num_classes)
        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # SCION_CHANGE:
            # n_params -= self.transformer.wpe.weight.numel()
            pass
        return n_params

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Semi-orthogonal initialization
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.Conv2d):
            # Also apply to convolution embedding if used
            print('/!\ conv')
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, x):
        has_nan = torch.isnan(x).any()
        has_inf = torch.isinf(x).any()
        if has_nan:
            print('nan in x before emb')
        if has_inf:
            print('inf in x before emb')
        out = self.patch_emb(x)
        has_nan = torch.isnan(out).any()
        has_inf = torch.isinf(out).any()
        if has_nan:
            print('nan after emb')
        if has_inf:
            print('inf after emb')
        if torch.isnan(out.mean()):
            print('after embedding:', out.mean())
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        
        # out = self.mixer_layers(out)
        for layer in self.mixer_layers:
            out = checkpoint(layer, out)
        
        ## batch norm:
        if self.normalization == "batchnorm":
            out = out.permute(0,2,1)
            out = self.ln(out)
            out = out.permute(0,2,1)
        # other
        else:
            out = self.ln(out)

        out = out[:, 0] if self.is_cls_token else out.mean(dim=1) # average over the patches

        has_nan = torch.isnan(out).any()
        has_inf = torch.isinf(out).any()
        if has_nan:
            print('nan before self.clf')
        if has_inf:
            print('inf before self.clf')

        out = self.clf(out)

        has_nan = torch.isnan(out).any()
        has_inf = torch.isinf(out).any()
        if has_nan:
            print('nan after clf')
        if has_inf:
            print('inf after clf')
        # out = out/self.hidden_size
        
        # has_nan = torch.isnan(out).any()
        # has_inf = torch.isinf(out).any()
        # if has_nan:
        #     print('nan after self.clf')
        # if has_inf:
        #     print('inf after self.clf')

        # print('clf weight: ', self.clf.weight)
        # has_nan = torch.isnan(self.clf.weight).any()
        # has_inf = torch.isinf(self.clf.weight).any()
        # if has_nan:
        #     print('nan in self.clf')
        # if has_inf:
        #     print('inf in self.clf')

        return out


class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act, normalization, convex_skip_connection, sqrt_convex_skip_connection, num_blocks):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act, normalization, convex_skip_connection, sqrt_convex_skip_connection, num_blocks)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act, normalization, convex_skip_connection, sqrt_convex_skip_connection, num_blocks)
    def forward(self, x):
        has_nan, has_inf = torch.isnan(x).any(), torch.isinf(x).any()
        if has_nan or has_inf:
            print(f"nan in x: {has_nan} | inf in x: {has_inf}")

        # x = torch.nan_to_num(x, posinf=25)

        
        out = self.mlp1(x)
        
        has_nan, has_inf = torch.isnan(out).any(), torch.isinf(out).any()
        if has_nan or has_inf:
            print(f"nan after mlp1: {has_nan} | inf after mlp1: {has_inf}")
            print(f"\nMLP1 fc1 Weight |  mean value: {torch.mean(self.mlp1.fc1.weight)} | max value:  {torch.max(self.mlp1.fc1.weight)}")
            print(f"\nMLP1 fc2 Weight |  mean value: {torch.mean(self.mlp1.fc2.weight)} | max value:  {torch.max(self.mlp1.fc2.weight)}")
        
        # out = torch.nan_to_num(out, posinf=25)

      
        out = self.mlp2(out)

        has_nan, has_inf = torch.isnan(out).any(), torch.isinf(out).any()
        if has_nan or has_inf:
            print(f"nan after mlp2: {has_nan} | inf after mlp2: {has_inf}")
            print(f"\nMLP2 fc1 Weight |  mean value: {torch.mean(self.mlp2.fc1.weight)} | max value:  {torch.max(self.mlp2.fc1.weight)}")
            print(f"\nMLP2 fc2 Weight |  mean value: {torch.mean(self.mlp2.fc2.weight)} | max value:  {torch.max(self.mlp2.fc2.weight)}")

        # out = torch.nan_to_num(out, posinf=25)
        
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act, normalization, convex_skip_connection, sqrt_convex_skip_connection, num_blocks):
        super(MLP1, self).__init__()
        norm_dict = {
            "layernorm": lambda hidden_size: nn.LayerNorm(hidden_size),
            "batchnorm": lambda hidden_size: nn.BatchNorm1d(hidden_size),
            "none": lambda hidden_size: nn.Identity(),
            "rmsnorm": lambda hidden_size: nn.RMSNorm(hidden_size, elementwise_affine=False)
        }
        self.normalization = normalization
        self.ln = norm_dict[normalization](hidden_size)

        self.convex_skip_connection = convex_skip_connection
        self.sqrt_convex_skip_connection = sqrt_convex_skip_connection
        self.num_blocks = num_blocks

        # self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.fc1 = nn.Linear(num_patches, hidden_s)
        self.do1 = nn.Dropout(p=drop_p)
        # self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.fc2 = nn.Linear(hidden_s, num_patches)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    
    def forward(self, x):
        if self.normalization == "batchnorm":
            out = x.permute(0,2,1)
            out = self.ln(out)
            out = out.permute(0,2,1)
        else:
            out = self.ln(x)

        # out = self.do1(self.act(self.fc1(out)))
        # out = self.do2(self.fc2(out))
        out = self.do1(self.act(self.fc1(out.transpose(1,2)).transpose(1,2)))
        out = self.do2(self.fc2(out.transpose(1,2)).transpose(1,2))

        if self.convex_skip_connection:
            factor = 1/math.sqrt(2*self.num_blocks) if self.sqrt_convex_skip_connection else 1/(2*self.num_blocks)
            return x + factor*out
        else:
            return out+x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act, normalization, convex_skip_connection, sqrt_convex_skip_connection, num_blocks):
        super(MLP2, self).__init__()
        norm_dict = {
            "layernorm": lambda hidden_size: nn.LayerNorm(hidden_size),
            "batchnorm": lambda hidden_size: nn.BatchNorm1d(hidden_size),
            "none": lambda hidden_size: nn.Identity(),
            "rmsnorm": lambda hidden_size: nn.RMSNorm(hidden_size, elementwise_affine=False)
        }
        self.normalization = normalization
        self.ln = norm_dict[normalization](hidden_size)

        self.convex_skip_connection = convex_skip_connection
        self.sqrt_convex_skip_connection = sqrt_convex_skip_connection
        self.num_blocks = num_blocks

        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    def forward(self, x):
        if self.normalization == "batchnorm":
            out = x.permute(0,2,1)
            out = self.ln(out)
            out = out.permute(0,2,1)
        else:
            out = self.ln(x)

        # print('after normalization:', out.mean())
        
        out = self.do1(self.act(self.fc1(out)))

        # print('after fc1:', out.mean())

        # if torch.isnan(self.fc2(out).mean()):
        #     print('in mlp2, after fc2:', self.fc2(out).mean())
        #     print('x:', out.mean())
        #     print('weigth fc2:', self.fc2.weight.mean())
        #     print('bias fc2:', self.fc2.bias.mean())

        #     print('out shape:', out.shape, ' | weight shape: ',  self.fc2.weight.shape, ' | bias shape: ', self.fc2.bias.shape)
        #     print('out * weight : ', (out.bfloat16()@self.fc2.weight.bfloat16().T ).mean(), (out@self.fc2.weight.T ).shape)

        #     sys.exit()

        
        # print('after self.fc2(out)', self.fc2(out).mean())
        # print("x range:", out.min().item(), out.max().item())
        # print("weight range:", self.fc2.weight.min().item(), self.fc2.weight.max().item())
        # print("bias range:", self.fc2.bias.min().item(),self.fc2.bias.max().item())

        out = self.do2(self.fc2(out))
        
        if self.convex_skip_connection:
            # print('mlp 2 convex sc')
            factor = 1/math.sqrt(2*self.num_blocks) if self.sqrt_convex_skip_connection else 1/(2*self.num_blocks)
            return x + factor*out
        else:
            return out+x

if __name__ == '__main__':
    net = MLPMixer(
        in_channels=3,
        img_size=32, 
        patch_size=4, 
        hidden_size=128, 
        hidden_s=512, 
        hidden_c=64, 
        num_layers=1, 
        num_classes=10, 
        drop_p=0.,
        off_act=False,
        is_cls_token=True
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice being used: {device}\n')
    net.to(device)
    print(net)
    torchsummary.summary(net, (3,32,32))
