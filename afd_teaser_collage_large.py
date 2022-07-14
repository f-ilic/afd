import argparse
from ast import Slice
import os
from os import listdir, mkdir
from os.path import isfile, join
from pathlib import Path

from matplotlib import patheffects
import flowiz as fz

import matplotlib.pyplot as plt
import torch
from torchvision.io import read_video, write_video
from torchvision.transforms import CenterCrop
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from SliceViewer import SliceViewer
from util import batch_warp, upsample_flow, warp
from generate_afd import flow_from_video
from util import batch_warp, firstframe_warp, upsample_flow, warp
parser = argparse.ArgumentParser()
import torchvision
torchvision.set_video_backend("video_reader")
from save_helper import save_tensor_as_img, save_tensor_list_as_gif
import matplotlib
matplotlib.use('Agg')

def main():
    parser.add_argument('--images', type=str, default=None, help='Path where to save individual images')

    args = parser.parse_args()
    args.alternate_corr = False
    args.mixed_precision = True
    args.small = False

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('RAFT/models/raft-sintel.pth'))
    model = model.module
    model.cuda()
    model.eval()

    classname = 'BenchPress'
    # classname= 'JumpingJack'
    video_paths = [
        f'/home/f/datasets/ucf101/{classname}/v_{classname}_g01_c02.avi',
        f'/home/f/datasets/ucf101/{classname}/v_{classname}_g02_c03.avi',
        f'/home/f/datasets/ucf101/{classname}/v_{classname}_g10_c01.avi',
        f'/home/f/datasets/ucf101/{classname}/v_{classname}_g03_c04.avi',
        f'/home/f/datasets/ucf101/{classname}/v_{classname}_g14_c01.avi',
    ]
    tmplist = []
    for video_path in video_paths:
        frames = (read_video(video_path)[0]).float()
        frame_list = list(frames.permute(0,3,1,2))

        flow = flow_from_video(model, frame_list)
        afd = firstframe_warp(flow, usecolor=True)

        afd = (CenterCrop(224)(afd.permute(0,3,1,2)) / 255.).permute(0,2,3,1)

        tmp = torch.cat([afd], dim=1)
        tmplist.append(tmp)

    min_frames = min([x.shape[0] for x in tmplist])
    shortened_clips = [x[:min_frames] for x in tmplist]
    tmp = torch.cat(shortened_clips, dim=2)

    if args.images != None:
        print("to make a high quality gif of the generated images:  gifski --fps 30 -o file.gif --quality 100 *.png")
        Path(join(args.images)).mkdir(parents=True, exist_ok=True)
        for i in range(tmp.shape[0]):
            fig, ax = plt.subplots(1,1, figsize=(len(video_paths)*6,6))
            ax.imshow(tmp[i])
            txt = plt.text(10,210, classname, size=50, color='white', family='serif')
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])
            plt.draw()
            plt.tight_layout()
            plt.axis('off')
            # plt.show()
            plt.savefig(f'{args.images}/{i:04d}.png', bbox_inches='tight', pad_inches = 0)


        # for i in range(tmp.shape[0]):
        #     save_tensor_as_img(tmp[i].permute(2,0,1), path=f'{args.images}/{i:04d}.png')

    SliceViewer(tmp, figsize=(15,5))

if __name__ == "__main__":
    main()




