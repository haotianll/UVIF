_base_ = [
    '../_base_/datasets/forgerynet_video.py',
    '../_base_/schedules/default_sgd_100k.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='UnifiedDetector',
    with_kwargs=False,
    backbone=dict(
        type='ResNetTSM',
        depth=101,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        pretrained='torchvision://resnet101',
        num_segments=32,
        is_shift=True,
        shift_div=8,
    ),
    head=dict(
        type='UnifiedHead',
        in_channels=2048,
        decoder_dict={
            'video': dict(
                type='VideoClsHead',
                num_classes=2,
                loss_module=dict(type='CrossEntropyLoss', loss_weight=1.0),
            ),
        },
        train_tasks=['video'],
        test_tasks=['video'],
    ),
)
