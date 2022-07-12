import argparse
import os
from os.path import join
from pathlib import Path
import cv2
import torch
import torch.nn as nn
from torchvision.io import read_video, write_video
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
from util import batch_warp, firstframe_warp, upsample_flow, warp
import torchvision
torchvision.set_video_backend("video_reader")

def flow_from_video(model, frames_list, upsample_factor=1):
    num_frames = len(frames_list)
    all_flows = []
    c,h,w = frames_list[0].shape
    with torch.no_grad():
        for i in range(num_frames - 1):
            image0 = frames_list[i].unsqueeze(0).cuda()
            image1 = frames_list[i + 1].unsqueeze(0).cuda()

            if upsample_factor != 1:
                image0 = nn.Upsample(scale_factor=upsample_factor, mode='bilinear')(image0)
                image1 = nn.Upsample(scale_factor=upsample_factor, mode='bilinear')(image1)

            padder = InputPadder(image0.shape)
            image0, image1 = padder.pad(image0, image1)
            _, flow_up = model(image0, image1, iters=12, test_mode=True)
            flow_output = padder.unpad(flow_up).detach().cpu().squeeze().permute(1, 2, 0)
            fl = flow_output.detach().cpu().squeeze()
            fl = upsample_flow(fl, h, w)
            all_flows.append(fl)
    return all_flows

def gather_videopaths_in_directory(directory, alphabetical_range=['A','Z']):
    video_list = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            curr_file = os.path.join(path, name)
            if curr_file.endswith('.avi'):
                first_letter_of_class = name.split('_')[1][0]
                if first_letter_of_class>=alphabetical_range[0] and first_letter_of_class<=alphabetical_range[1]:
                    video_list.append(curr_file)
    return video_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', type=str, default='/home/f/datasets/ucf101', help='Path to UCF101 (e.g. "/home/f/datasets/ucf101")')
    parser.add_argument('--dst', type=str, default='./afd101test/', help='Where to save AFD101 (e.g. "/home/f/datasets/afd101")')
    parser.add_argument('--grayscale', type=bool, default=False, help='Default: make AFD in color (3 channel), grayscale==True makes it 1 channel')
    parser.add_argument('--start_letter', type=str, default='A')
    parser.add_argument('--end_letter', type=str, default='Z')
    parser.add_argument('-images', action='store_true',  help='Saves the dataset as individual images, instead of mp4')

    args = parser.parse_args()
    args.alternate_corr = False
    args.mixed_precision = True
    args.small = False

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('RAFT/models/raft-sintel.pth'))
    model = model.module
    model.cuda()
    model.eval()

    src_path = args.src
    dst_path = args.dst

    start_letter = args.start_letter
    end_letter = args.end_letter

    video_list = sorted(gather_videopaths_in_directory(src_path, (start_letter, end_letter)))
    for video_name in video_list:        
        action = video_name.split('/')[-2]
        video_id = video_name.split('/')[-1]
        
        if args.images:
            images_dir = video_id.split('.avi')[0]
            Path(join(dst_path, action, images_dir)).mkdir(parents=True, exist_ok=True)
            save_path = join(dst_path, action, images_dir)
        else:
            Path(join(dst_path, action)).mkdir(parents=True, exist_ok=True)
            save_path = join(dst_path, action, video_id)


        frames = (read_video(video_name)[0]).float()

        video = torchvision.io.VideoReader(video_name, 'video')
        fps = video.get_metadata()['video']['fps'][0]

        frame_list = list(frames.permute(0,3,1,2))
        
        flow = flow_from_video(model, frame_list)
        afd = firstframe_warp(flow, usecolor=(not args.grayscale))

        if os.path.isfile(save_path):
            print(f"[SKIP] {save_path}")
        else:
            if args.images:
                save_tensorvideo_as_images(save_path, afd)
            else:
                write_video(save_path, afd, fps=fps)
            print(f"[ OK ] {save_path}")


def save_tensorvideo_as_images(path, video):
    for t in range(video.shape[0]):
        img = (video[t,...] * 255).byte()
        total_path = f'{path}/{t:04d}.png'
        cv2.imwrite(total_path, img.numpy(), [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == "__main__":
    main()



