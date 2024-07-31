_base_ = [
    '../_base_/datasets/dfdcp_video_image.py',
    '../_base_/schedules/default_sgd_20k.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='UnifiedDetector',
    backbone=dict(
        type='ResNetTSM',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        pretrained='torchvision://resnet50',
        num_segments=32,
        is_shift=True,
        shift_div=8,
    ),
    head=dict(
        type='UnifiedHead',
        in_channels=2048,
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
