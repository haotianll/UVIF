# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=1e-4),
)

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, end=2000),
    dict(type='OneCycleLR', total_steps=100000, by_epoch=False, eta_max=1e-2),
]

# train, val, test setting
train_cfg = dict(by_epoch=False, max_iters=100000, val_interval=100000)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR, based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=16)
