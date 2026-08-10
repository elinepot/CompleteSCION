import torch
import torch.nn as nn
from spectral_clip import jiacheng_orthogonalize
from scion import zeropower_via_newtonschulz5

def normalize_model(net, radiuses, newton_schulz = 'jiacheng', eps=0.0):
    with torch.no_grad():
        idx = 0 
        for module in net.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                radius = radiuses[idx] - eps
                
                W = module.weight.data

                if isinstance(module, nn.Linear):
                    if newton_schulz=='jordan':
                        module.weight.data = radius*zeropower_via_newtonschulz5(W)
                    else :
                        module.weight.data = radius*jiacheng_orthogonalize(W)

                elif isinstance(module, nn.Conv2d):
                    W_reshaped = W.view(W.size(0), -1)  # (out_channels, -)
                    if newton_schulz=='jordan':
                        module.weight.data = (radius*zeropower_via_newtonschulz5(W_reshaped)).view_as(W)
                    else :
                        module.weight.data = (radius*jiacheng_orthogonalize(W_reshaped)).view_as(W)             
                
                if module.bias is not None:
                    module.bias.zero_()

                idx+=1
    return net