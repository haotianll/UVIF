# UVIF: Unified Video and Image Representation for Boosted Video Face Forgery Detection

This repository is the official implementation of paper: [Unified Video and Image Representation for Boosted Video Face Forgery Detection](https://doi.org/10.3233/FAIA240548)

## Environment

- python: 3.9
- pytorch: 2.0
- mmpretrain: 1.0.0rc8

```sh
conda create -n uvif python=3.9
conda activate uvif

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

pip install -U openmim

cd UVIF
mim install -e .
```

## Dataset Preparation

We perform experiments on [ForgeryNet](https://yinanhe.github.io/projects/forgerynet.html#)
and [DFDC (preview)](https://arxiv.org/abs/1910.08854).

We use [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) to detect face bounding boxes from each video clip
or static image. The processed json annotations with face regions are available at
this [link](https://drive.google.com/drive/folders/1ukB8oTH8enmgtcgNtzaqBj7Hdx3xHGq7).

The dataset directory structure is like this:

```
data/
├── ForgeryNet/
│   ├── annotations/
│   │   ├── video_train.json
│   │   └── ...
│   ├── Training/
│   │   ├── video
│   │   ├── image
│   │   └── ...
│   └── Validation/
│       ├── video
│       └── ...
└── DFDCP/
    ├── annotations/
    │   ├── train.json
    │   └── ...
    ├── method_A
    ├── method_B
    └── original_videos
```

## Training and Evaluation

Taking ForgeryNet as an example:

### Training

```shell
# baseline
bash tools/dist_train.sh configs_uvif/forgerynet/video_r50_forgerynet.py 2

# uvif
bash tools/dist_train.sh configs_uvif/forgerynet/uvif_r50_forgerynet.py 2
```

### Evaluation

```shell
# single gpu
python test.py configs_uvif/forgerynet/uvif_r50_forgerynet.py pretrained/uvif_r50_forgerynet.pth

# multiple gpu
bash tools/dist_test.sh configs_uvif/forgerynet/uvif_r50_forgerynet.py pretrained/uvif_r50_forgerynet.pth 2
```

### Results and models

The pretrained weights of the following models are available at
this [link](https://drive.google.com/drive/folders/1XrutVt4bVFGoMm3kmQvZ0yG3PN5fcGb2).

#### ForgeryNet

| method                | config                         | mAcc  | AUC   |
| --------------------- | ------------------------------ | ----- | ----- |
| Baseline - Res50      | video_r50_forgerynet.py        | 80.89 | 88.66 |
| Baseline - Res101     | video_r101_forgerynet.py       | 81.48 | 88.08 |
| Baseline - ConvNeXt-T | video_convnext-t_forgerynet.py | 81.56 | 88.43 |
| UVIF - Res50          | uvif_r50_forgerynet.py         | 85.32 | 93.45 |
| UVIF - Res101         | uvif_r101_forgerynet.py        | 86.57 | 94.42 |
| UVIF - ConvNeXt-T     | uvif_convnext-t_forgerynet.py  | 84.94 | 93.35 |

#### DFDC (preview)

| method        | config             | Acc   | AUC   |
| ------------- | ------------------ | ----- | ----- |
| UVIF - Res50  | uvif_r50_dfdcp.py  | 83.40 | 93.54 |
| UVIF - Res101 | uvif_r101_dfdcp.py | 87.00 | 94.95 |

## Acknowledgment

The code is based on [MMPretrain](https://github.com/open-mmlab/mmpretrain) and [MMAction2](https://github.com/open-mmlab/mmaction2). Thanks for their contributions.

## Citation

If you find this repository useful in your research, please consider citing:

```latex
@inproceedings{liu2024uvif,
  title={Unified Video and Image Representation for Boosted Video Face Forgery Detection},
  author={Liu, Haotian and Pan, Chenhui and Liu, Yang and Zhao, Guoying and Li, Xiaobai},
  booktitle={ECAI},
  pages={673-680},
  year={2024},
}
```
