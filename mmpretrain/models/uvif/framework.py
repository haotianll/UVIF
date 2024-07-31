from typing import List, Optional, Tuple, Sequence

import torch
import torch.nn as nn
from mmengine.model import BaseModel, BaseModule, ModuleDict
from mmengine.structures import BaseDataElement as DataSample

from mmpretrain.registry import MODELS
from mmpretrain.structures.unified_data_sample import UnifiedDataSample


@MODELS.register_module()
class UnifiedDetector(BaseModel):
    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 with_temporal=True,
                 with_kwargs=True,
                 ):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got {type(data_preprocessor)}')

        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.with_temporal = with_temporal
        self.with_kwargs = with_kwargs

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor',
                **kwargs):

        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat_origin(self, x):
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        out = x
        return tuple(out)

    def extract_feat_image(self, image):
        if self.with_kwargs:
            image = self.backbone(image, with_temporal=False)
        else:
            image = self.backbone(image)

        if self.with_neck:
            image = self.neck(image)

        return tuple(image)

    def extract_feat_video(self, video):
        bs = video.shape[0]
        video = video.view((-1,) + video.shape[2:])

        if self.with_kwargs:
            video = self.backbone(video, with_temporal=self.with_temporal)
        else:
            video = self.backbone(video)

        if self.with_neck:
            video = self.neck(video)

        if isinstance(video, tuple) and len(video[0].shape) == 4:
            video = tuple([i.reshape((bs, -1) + i.shape[1:]) for i in video])
        return video

    def extract_feat(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            out = dict()
            if 'image' in inputs.keys():
                out['image'] = self.extract_feat_image(inputs['image'])

            if 'video' in inputs.keys():
                if not isinstance(inputs['video'], List):
                    out['video'] = self.extract_feat_video(inputs['video'])
                else:
                    video = inputs['video'][0]
                    video_weak = inputs['video'][1]

                    out['video'] = self.extract_feat_video(video)
                    with torch.no_grad():
                        out['video_weak'] = self.extract_feat_video(video_weak)
        else:
            out = self.extract_feat_origin(inputs)
        return out

    def extract_feats(self, multi_inputs: Sequence[torch.Tensor], **kwargs) -> list:
        assert isinstance(multi_inputs, Sequence), \
            '`extract_feats` is used for a sequence of inputs tensor. If you ' \
            'want to extract on single inputs tensor, use `extract_feat`.'
        return [self.extract_feat(inputs, **kwargs) for inputs in multi_inputs]

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample], **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every sample.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples, **kwargs)

    def train_step(self, data, optim_wrapper):
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars


@MODELS.register_module()
class UnifiedHead(BaseModule):
    def __init__(self,
                 in_channels,
                 in_index=-1,
                 decoder_dict: dict = dict(),
                 auxiliary_dict: dict = dict(),
                 init_cfg: Optional[dict] = None,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),

                 train_tasks=['video', 'image'],
                 test_tasks=['video'],
                 task_types={
                     'image': 'image',
                     'video': 'video',
                 },
                 ):
        super().__init__(init_cfg=init_cfg)

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        self.task_types = task_types

        self.decoder_modules = ModuleDict()
        self.auxiliary_modules = ModuleDict()

        for task_name, decoder_module in decoder_dict.items():
            if task_name not in self.train_tasks:
                raise NotImplementedError(task_name)

            decoder_module['in_channels'] = in_channels

            self.decoder_modules.register_module(task_name, MODELS.build(decoder_module))

        for task_name, auxiliary_module in auxiliary_dict.items():
            if auxiliary_module is None or task_name not in self.train_tasks:
                continue
            self.auxiliary_modules.register_module(task_name, MODELS.build(auxiliary_module))

    def forward(self, feats: Tuple[torch.Tensor], task_names=None) -> torch.Tensor:
        pred_results = dict()

        for task_name, decoder in self.decoder_modules.items():
            if task_names is not None and task_name not in task_names:
                continue

            key = self.task_types[task_name]
            _feats = feats[key]
            pred_results[key] = decoder(_feats)

        return pred_results

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample], **kwargs) -> dict:
        pred_results = self(feats, task_names=self.train_tasks)

        losses = dict()
        for task_name, decoder in self.decoder_modules.items():
            if task_name not in self.train_tasks:
                continue

            key = self.task_types[task_name]
            pred = pred_results[key]
            sub_data_samples = data_samples[key]

            loss = decoder.loss(pred, sub_data_samples, task_name=task_name, **kwargs)
            losses.update(loss)

        for name, auxiliary_module in self.auxiliary_modules.items():
            loss = auxiliary_module.loss(feats, data_samples, self.decoder_modules, **kwargs)
            losses.update(loss)

        return losses

    def predict(self,
                feats: Tuple[torch.Tensor],
                data_samples: Optional[List[Optional[UnifiedDataSample]]] = None) -> List[UnifiedDataSample]:
        pred_results = self(feats, task_names=self.test_tasks)

        if data_samples is None:
            raise NotImplementedError()

        for task_name, decoder in self.decoder_modules.items():
            if task_name not in self.test_tasks:
                continue

            key = self.task_types[task_name]
            pred = pred_results[key]
            _data_samples = data_samples[key]

            _data_samples = decoder.predict(pred, _data_samples, task_name=task_name)
            data_samples[key] = _data_samples

        return data_samples
