import argparse
from ast import Slice
import os
from os import listdir, mkdir
from os.path import isfile, join
from pathlib import Path
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

def main():
    parser.add_argument('--images', type=str, default=None, help='Path where to save individual images')
    parser.add_argument('--upsample', type=int, default=1, help='Upsample image factor, if image too small to calculate optical flow reliably')

    args = parser.parse_args()
    args.alternate_corr = False
    args.mixed_precision = True
    args.small = False

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('RAFT/models/raft-sintel.pth'))
    model = model.module
    model.cuda()
    model.eval()

    # collage 1
    # save_path='collage1.gif'
    # video_paths = [
    #     '/home/f/datasets/ucf101/PushUps/v_PushUps_g01_c02.avi',
    #     '/home/f/datasets/ucf101/HeadMassage/v_HeadMassage_g02_c03.avi',
    #     '/home/f/datasets/ucf101/PommelHorse/v_PommelHorse_g02_c03.avi'
    # ]
    # collage 2
    save_path='collage2.gif'
    video_paths = [
        '/home/f/datasets/ucf101/TrampolineJumping/v_TrampolineJumping_g03_c01.avi',
        '/home/f/datasets/ucf101/Archery/v_Archery_g01_c01.avi',
        '/home/f/datasets/ucf101/Knitting/v_Knitting_g02_c02.avi',
    ]
    tmplist = []
    for video_path in video_paths:
        frames = (read_video(video_path)[0]).float()
        frame_list = list(frames.permute(0,3,1,2))

        video = torchvision.io.VideoReader(video_path, 'video')
        fps = video.get_metadata()['video']['fps'][0]
        frame_duration_ms =  40 # adjust this so it looks good for the combination of the videos

        flow = flow_from_video(model, frame_list, upsample_factor=args.upsample)
        afd = firstframe_warp(flow, usecolor=True)

        frames = (CenterCrop(224)(frames.permute(0,3,1,2)) / 255.).permute(0,2,3,1)
        afd = (CenterCrop(224)(afd.permute(0,3,1,2)) / 255.).permute(0,2,3,1)

        tmp = torch.cat([frames, afd], dim=1)
        tmplist.append(tmp)

    min_frames = min([x.shape[0] for x in tmplist])
    shortened_clips = [x[:min_frames] for x in tmplist]
    tmp = torch.cat(shortened_clips, dim=2)
    save_tensor_list_as_gif(list(tmp.permute(0,3,1,2)), path=save_path, duration=frame_duration_ms)

    if args.images != None:
        print("to make a high quality gif of the generated images:  gifski --fps 30 -o file.gif --quality 100 *.png")
        Path(join(args.images)).mkdir(parents=True, exist_ok=True)
        for i in range(tmp.shape[0]):
            save_tensor_as_img(tmp[i].permute(2,0,1), path=f'{args.images}/{i:04d}.png')

    SliceViewer(tmp, figsize=(15,5))

if __name__ == "__main__":
    main()




