dependencies = ['torch']
from zero_dcepp import enhance_net_nopool
import torch

def DCE_net(scale_factor=1):
    _DCE_net = enhance_net_nopool(scale_factor)
    checkpoint = 'https://github.com/ernestchu/Zero-DCE_extension/releases/download/0.0.1/Epoch99.pth'
    _DCE_net.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint))

    return _DCE_net
