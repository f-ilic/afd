
*Disclaimer:* This repository is a (standalone) submodule of https://github.com/f-ilic/AppearanceFreeActionRecognition. This repo only contains the dataset  and dataset related code.

---

**Is Appearance Free Action Recognition Possible?**<br>
ECCV 2022<br>
Filip Ilic, Thomas Pock, Richard P. Wildes<br>

---

[[Project Page]](https://f-ilic.github.io/AppearanceFreeActionRecognition)
[[Paper]](https://arxiv.org/pdf/2207.06261.pdf)

Environment Setup:
```bash
git clone git@github.com:f-ilic/afd.git
conda env create -f environment.yaml
conda activate motion
```



# Appearance Free Dataset
The Appearance Free Dataset (AFD) is an extension to a widely used action recognition dataset that
disentangles static and dynamic video components. In particular, no single frame
contains any static discriminatory information in AFD; the action is only encoded in the temporal dimension.

We provide the AFD101 dataset based on UCF101, and also the necessary tools to generate other appearance free videos.

Some samples from AFD101 are shown below:
![](readme_assets/collage1.gif)


![](readme_assets/collage2.gif)


![](readme_assets/afdjumpingjack.gif)
![](readme_assets/afdbenchpress.gif)


## Download
Download link to AFD101: https://files.icg.tugraz.at/f/9b501254b9b0499cbec9/


If you wish to download the whole dataset package, run:
```bash
wget https://files.icg.tugraz.at/f/b751f3b2a53440039d89/?dl=1 -O ucf101.zip
wget https://files.icg.tugraz.at/f/9b501254b9b0499cbec9/?dl=1 -O afd101.zip
unzip ucf101.zip
unzip afd101.zip
```

[ALTERNATE AFD101 MIRROR](https://cloud.tugraz.at/index.php/s/BMfNRLPLsXs5qsM)

## Create it yourself

*Note: To run the following scripts `cd` into `afd/`*


To create it from scratch you need to

1) Download and extract ucf101

```
wget https://files.icg.tugraz.at/f/b751f3b2a53440039d89/?dl=1 -O ucf101.zip
unzip ucf101
```

2) Run:
```python 
python generate_afd.py --src /path/to/ucf101/ --dst /some/path/
```

**Optional:**

If you want to save the individual frames instead, which leads to higher fidelity images due to the compression induced from mp4 and the fact that random noise is hard to compress you can pass the `-images` flag. Be aware that this will use around 300gb of space. This was <u>not</u> used in the paper.

```python 
python generate_afd.py --src /path/to/ucf101/ --dst /some/path/ -images
```

Both methods will populate `/some/path` with the AFD101 dataset.

*Hint:* For quick speedup if you have multiple GPUs:
```
CUDA_VISIBLE_DEVICES=0 python generate_afd.py --src /home/data/ucf101 --dst /home/data/afd101 --start_letter A --end_letter F
CUDA_VISIBLE_DEVICES=1 python generate_afd.py --src /home/data/ucf101 --dst /home/data/afd101 --start_letter F --end_letter J
CUDA_VISIBLE_DEVICES=2 python generate_afd.py --src /home/data/ucf101 --dst /home/data/afd101 --start_letter J --end_letter P
CUDA_VISIBLE_DEVICES=3 python generate_afd.py --src /home/data/ucf101 --dst /home/data/afd101 --start_letter P --end_letter Z
```

To generate the optical flow datasets needed for on demand dataloading run:
```python
python generate_flow.py --src /path/to/ucf101orafd101/ --dst /some/path/
```

 ## Visualisation & Utilities

We also provide some basic utilities to view AFD data in context of its optical flow and the (regular rgb) video. 
This is useful to see the streghts and weaknesses of AFD video.



 To visualize a (regular) video side by side of an appearance free version and the middleburry coded optical flow of said video, run:
 ```python
python afd_single_video.py /path/to/video.mp4
 ```

![](readme_assets/demo_afd_single_video.gif)

(click on gif or zoom in to get rid of the moreau pattern)


To save the result as a gif (as seen above):

 ```python
python afd_single_video.py /path/to/video.mp4 --gif /home/Desktop/test.gif
 ```


Furthermore if you want to extract AFD from already AFD video (😳) passt the flag `--upsample INT` so that RAFT can find correspondances more easily as 320x240 images are too small with /4 downscaling in RAFT. You can also try this if your videos have a small spatial resolution.

 ```python
python afd_single_video.py /path/to/afdvideo.mp4 --gif /home/Desktop/test.gif --upsample 2
 ```

 To create a collage of a few AFD videos as shown on top run:

 ```python 
 python afd_teaser_collage.py
 ```
 The output will be saved as `collage.gif`.

# Acknowledgements

Parts of this project use code from the following repositories:

* https://github.com/princeton-vl/RAFT
* https://github.com/georgegach/flowiz


# Cite
```bibtex
@InProceedings{ilic22appearancefree,
title={Is Appearance Free Action Recognition Possible?},
author={Ilic, Filip and Pock, Thomas and Wildes, Richard P.},
booktitle={European Conference on Computer Vision (ECCV)},
month={October},
year={2022},
}
```
