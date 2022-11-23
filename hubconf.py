dependencies = ['torch']
from zero_dcepp import enhance_net_nopool
import torch

def DCE_net(scale_factor=1):
    _DCE_net = enhance_net_nopool(scale_factor)
    _DCE_net.load_state_dict(torch.load('zero_dcepp/weights/Epoch99.pth'))

    return _DCE_net
