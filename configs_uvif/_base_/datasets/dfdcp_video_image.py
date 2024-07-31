# dataset settings
data_root = 'data/'

ann_files = [
    'DFDCP/annotations/train.json',
    'DFDCP/annotations/test.json',
    'ForgeryNet/annotations/image_train_sub100k.json',
]

data_prefixes = [
    dict(video_path='DFDCP'),
    dict(video_path='DFDCP'),
    dict(img_path='ForgeryNet/Training/image'),
]

batch_size_video = 8
batch_size_image = 64

num_gpus = 2

data_preprocessor = dict(
    type='UnifiedDataPreprocessor',
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

albu_transform_list = [
    dict(type='GaussNoise', p=1.0, var_limit=(10.0, 50.0), per_channel=True, mean=0),
    dict(type='Sharpen', p=1.0, alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
    dict(type='RandomBrightnessContrast',
         brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), brightness_by_max=True, p=1.0),
    dict(type='ImageCompression', quality_lower=1, quality_upper=99, p=1.0),
    dict(type='GaussianBlur', blur_limit=(3, 11), p=1.0),
    dict(type='CLAHE', clip_limit=(1, 8), tile_grid_size=(8, 8), p=1.0),
    dict(type='RandomGamma', gamma_limit=(10, 150), eps=None, p=1.0),
    dict(type='ToGray', p=1.0),
    dict(type='ChannelShuffle', p=1.0),
]

albu_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='SomeOf', transforms=albu_transform_list, n=2, p=1.0),
                    dict(type='SomeOf', transforms=albu_transform_list, n=3, p=1.0),
                    dict(type='SomeOf', transforms=albu_transform_list, n=4, p=1.0),
                ], p=0.98
            ),
            dict(type='OneOf', transforms=albu_transform_list, p=0.01)
        ], p=0.99
    )
]

albu_transforms_video = [
    dict(
        type='Compose',
        transforms=albu_transforms,
        additional_targets={f'image{i}': 'image' for i in range(0, batch_size_image)}
    )
]

train_pipeline_video = [
    dict(type='VideoDecordInit', io_backend='disk'),
    dict(type='VideoSampleFrames', clip_len=32, frame_interval=4, num_clips=1),
    dict(type='VideoDecordDecode'),
    dict(type='VideoFaceCrop'),
    dict(type='VideoRandomResize', scale=(224, 224), ratio_range=(1., 8. / 7.)),
    dict(type='VideoRandomCrop', crop_size=224),
    dict(type='VideoRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='VideoMultiView',
         num_views=[1, 1],
         transforms=[
             [dict(type='VideoAlbu', transforms=albu_transforms_video)],
             [dict(type='VideoIdentity')]
         ]),
    dict(type='VideoFormatShape', input_format='NCHW'),
    dict(type='PackVideoInputs'),
]

test_pipeline_video = [
    dict(type='VideoDecordInit', io_backend='disk'),
    dict(type='VideoSampleFrames', clip_len=32, frame_interval=4, num_clips=1, test_mode=True),
    dict(type='VideoDecordDecode'),
    dict(type='VideoFaceCrop'),
    dict(type='VideoResize', scale=224),
    dict(type='VideoFormatShape', input_format='NCHW'),
    dict(type='PackVideoInputs'),
]

train_pipeline_image = [
    dict(type='LoadImageFromFile'),
    dict(type='FaceCrop'),
    dict(type='RandomResize', scale=(224, 224), ratio_range=(1., 8. / 7.)),
    dict(type='RandomCrop', crop_size=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Albu', transforms=albu_transforms),
    dict(type='PackImageInputs'),
]

train_dataset_video = dict(
    type='UnifiedVideoDataset',
    data_root=data_root,
    ann_file=ann_files[0],
    data_prefix=data_prefixes[0],
    pipeline=train_pipeline_video
)

test_dataset_video = dict(
    type='UnifiedVideoDataset',
    data_root=data_root,
    ann_file=ann_files[1],
    data_prefix=data_prefixes[1],
    pipeline=test_pipeline_video
)

train_dataset_image = dict(
    type='UnifiedImageDataset',
    data_root=data_root,
    ann_file=ann_files[2],
    data_prefix=data_prefixes[2],
    pipeline=train_pipeline_image
)

train_dataloader = dict(
    batch_size=batch_size_image + batch_size_video,
    num_workers=8,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            train_dataset_image,
            train_dataset_video,
        ]
    ),
    sampler=dict(
        type='MultiSourceSampler',
        batch_size=batch_size_image + batch_size_video,
        source_ratio=[batch_size_image, batch_size_video]
    ),
    collate_fn=dict(type='pseudo_collate'),
)

val_dataloader = dict(
    batch_size=batch_size_video * 2,
    num_workers=8,
    dataset=test_dataset_video,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='pseudo_collate'),
)

val_evaluator = dict(
    type='UnifiedEvaluator',
    metrics=dict(
        type='UnifiedMetric',
        task_metrics={
            'video': [
                dict(type='Accuracy', topk=(1,)),
                dict(type='AUC'),
            ],
        }
    )
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator
