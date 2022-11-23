dependencies = ['torch']
from zero_dcepp import enhance_net_nopool
import torch

def DCE_net(scale_factor=1):
    _DCE_net = enhance_net_nopool(scale_factor)
    checkpoint = 'To-be-added'
    _DCE_net.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint))

    return _DCE_net
