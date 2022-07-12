import argparse
from ast import Slice
import os
from os import listdir, mkdir
from os.path import isfile, join
from pathlib import Path
import flowiz as fz
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_video, write_video
from torchvision.transforms import CenterCrop
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from SliceViewer import SliceViewer
from util import batch_warp, upsample_flow, warp
from generate_afd import flow_from_video, firstframe_warp
from matplotlib import patheffects, transforms

parser = argparse.ArgumentParser()
import torchvision
torchvision.set_video_backend("video_reader")
from save_helper import save_tensor_list_as_gif

def main():

    args = parser.parse_args()
    args.alternate_corr = False
    args.mixed_precision = True
    args.small = False

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('afd/RAFT/models/raft-sintel.pth'))
    model = model.module
    model.cuda()
    model.eval()

    # collage 1
    save_path='teaser1.png'
    video_path='/home/f/datasets/ucf101/PlayingViolin/v_PlayingViolin_g10_c03.avi'
    frame_number_to_save = 15


    save_path='teaser2.png'
    video_path='/home/f/datasets/ucf101/Knitting/v_Knitting_g11_c04.avi'
    frame_number_to_save = 41

    classname = video_path.split("/")[-2]


    frames = (read_video(video_path)[0]).float()
    frame_list = list(frames.permute(0,3,1,2))

    flow = flow_from_video(model, frame_list)
    afd = firstframe_warp(flow, usecolor=True)

    frames = (CenterCrop(224)(frames.permute(0,3,1,2)) / 255.).permute(0,2,3,1)
    afd = (CenterCrop(224)(afd.permute(0,3,1,2)) / 255.).permute(0,2,3,1)
    rgbflow = (CenterCrop(224)(torch.from_numpy((fz.video_flow2color(fz.video_normalized_flow(torch.stack(flow,dim=0))))).permute(0,3,1,2))).permute(0,2,3,1)

    tmp = torch.cat([frames[:-1], rgbflow/255., afd[:-1]], dim=2)
    SliceViewer(tmp, figsize=(15,5))
    fig, ax = plt.subplots(1,1, figsize=(10,3.5))
    ax.imshow(tmp[frame_number_to_save])
    txt = plt.text(10,210, classname, size=25, color='white', family='serif')
    txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])
    plt.draw()
    plt.tight_layout()
    plt.axis('off')
    # plt.show()
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)





if __name__ == "__main__":
    main()




