_base_ = [
    '../_base_/datasets/forgerynet_video_image.py',
    '../_base_/schedules/default_adamw_100k.py',
    '../_base_/default_runtime.py',
]

# https://github.com/open-mmlab/mmpretrain/tree/main/configs/convnext
pretrained = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth'

model = dict(
    type='UnifiedDetector',
    backbone=dict(
        type='ConvNeXtTSM',
        arch='tiny',
        drop_path_rate=0.1,
        num_segments=32,
        is_shift=True,
        shift_div=8,
        init_cfg=dict(
            type='PretrainedTSM', prefix='backbone', checkpoint=pretrained,
            revise_keys=[('.depthwise_conv.', '.depthwise_conv.module.')]
        ),
    ),
    head=dict(
        type='UnifiedHead',
        in_channels=768,
        decoder_dict={
            'video': dict(
                type='VideoClsHeadWithFrames',
                num_classes=2,
                loss_module=dict(type='CrossEntropyLoss', loss_weight=1.0),
            ),
            'image': dict(
                type='ImageClsHead',
                num_classes=2,
                loss_module=dict(type='CrossEntropyLoss', loss_weight=1.0)
            ),
        },
        auxiliary_dict={
            'pseudo': dict(
                type='VideoPseudoLabeling',
                loss_pseudo_module=dict(type='CrossEntropyLoss', loss_weight=1.0),
            ),
        },
        train_tasks=['video', 'image', 'pseudo'],
        test_tasks=['video'],
    ),
)
