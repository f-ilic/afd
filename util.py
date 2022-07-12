from os import listdir
from os.path import join, isfile

import torch
from torch.nn.functional import grid_sample
from torchvision.transforms import PILToTensor, Resize
from PIL import Image
import numpy as np
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

def warp(img, flow, mode='bilinear', padding_mode='zeros'):
    # img.shape -> 3, H, W
    # flow.shape -> H, W, 2
    _, h, w, = img.shape
    y_coords = torch.linspace(-1, 1, h)
    x_coords = torch.linspace(-1, 1, w)
    f0 = torch.stack(torch.meshgrid(x_coords, y_coords)).permute(2, 1, 0)

    f = f0 + torch.stack([2 * (flow[:, :, 0] / w),
                          2 * (flow[:, :, 1] / h)], dim=2)
    warped = grid_sample(img.unsqueeze(0), f.unsqueeze(0), mode=mode, padding_mode=padding_mode)
    return warped.squeeze()


def batch_warp(img, flow, mode='bilinear', padding_mode='zeros'):
    # img.shape -> B, 3, H, W
    # flow.shape -> B, H, W, 2
    b, c, h, w, = img.shape
    y_coords = torch.linspace(-1, 1, h)
    x_coords = torch.linspace(-1, 1, w)
    f0 = torch.stack(torch.meshgrid(x_coords, y_coords)).permute(2, 1, 0).repeat(b,1,1,1)

    f = f0 + torch.stack([2 * (flow[..., 0] / w),
                          2 * (flow[..., 1] / h)], dim=3)
    warped = grid_sample(img, f, mode=mode, padding_mode=padding_mode)
    return warped.squeeze()

def upsample_flow(flow, h, w):
    # usefull function to bring the flow to h,w shape
    # so that we can warp effectively an image of that size with it
    h_new, w_new, _ = flow.shape
    flow_correction = torch.Tensor((h / h_new, w / w_new))
    f = (flow * flow_correction[None, None, :])

    f = (Resize((h, w), interpolation=InterpolationMode.BICUBIC)(f.permute(2, 0, 1))).permute(1, 2, 0)
    return f


def firstframe_warp(flows, interpolation_mode='nearest', usecolor=True):
    h, w, _ = flows[0].shape
    c = 3 if usecolor==True else 1

    t = len(flows)
    flows = torch.stack(flows)
    
    if usecolor:
        inits = (torch.rand((1, c, h, w))).repeat(t, 1, 1, 1)
    else:
        inits = (torch.rand((1, 1, h, w))).repeat(t, 3, 1, 1)

    warped = batch_warp(inits, flows, mode=interpolation_mode)
    masks = (~(warped.any(dim=1)))
    masks = masks.unsqueeze(1).repeat(1, 3, 1, 1)
    warped[masks] = inits[masks]
    warped = torch.cat([inits[0,...].unsqueeze(dim=0), warped], dim=0)
    warped = (warped).clip(0, 1)
    warped = (warped.permute(0,2,3,1) * 255).float()
    return warped


def continuous_warp(flows, interpolation_mode='nearest', usecolor=True):
    h, w, _ = flows[0].shape
    t = len(flows)
    c = 3 if usecolor==True else 1
    
    init = (torch.rand((c, h, w)))
    warp_i = init
    l=[]
    for en, flow in enumerate(flows):
        warp_i = warp(warp_i, flow, mode=interpolation_mode, padding_mode='reflection')
        if c==1:
            warp_i = warp_i.unsqueeze(0)

        mask = (~(warp_i.any(dim=0)))
        mask = mask.repeat(c, 1, 1)
        warp_i[mask] = init[mask]
        l.append(warp_i)

    warped = (torch.stack(l, axis=0))
    warped = (warped).clip(0, 1)
    warped = (warped.permute(0,2,3,1) * 255).float()
    return warped