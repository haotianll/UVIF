import copy
import inspect
from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn

from mmpretrain.registry import MODELS


def get_str_type(module) -> str:
    """Return the string type name of module.

    Args:
        module (str | ModuleType | FunctionType):
            The target module class

    Returns:
        Class name of the module
    """
    if isinstance(module, str):
        str_type = module
    elif inspect.isclass(module) or inspect.isfunction(module):
        str_type = module.__name__
    else:
        return None

    return str_type


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Defaults to 1.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class BaseVideoHead(BaseModule, metaclass=ABCMeta):

    def __init__(self,
                 loss_module: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 average_clips: str = 'prob',
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        if not isinstance(loss_module, nn.Module):
            loss_module = MODELS.build(loss_module)
        self.loss_module = loss_module

        self.average_clips = average_clips
        self.num_segs = None

    @abstractmethod
    def forward(self, inputs):
        pass

    def loss(self, cls_score, data_samples=None, target=None, task_name=None, **kwargs):
        if target is None:
            if data_samples is None:
                raise ValueError(data_samples)

            if 'gt_score' in data_samples[0].tasks(task_name):
                target = torch.stack([i.tasks(task_name).gt_score for i in data_samples])
            else:
                target = torch.cat([i.tasks(task_name).gt_label for i in data_samples])

        losses = dict()
        loss = self.loss_module(cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses[f'loss_{task_name}'] = loss
        return losses

    def predict(self, pred, data_samples, task_name=None, **kwargs):

        num_segs = pred.shape[0] // len(data_samples) if self.num_segs is None else self.num_segs
        pred_scores = self.average_clip(pred, num_segs=num_segs)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        for data_sample, score, label in zip(data_samples, pred_scores, pred_labels):
            sub_data_sample = data_sample.tasks(task_name)
            sub_data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples

    def average_clip(self,
                     cls_scores: torch.Tensor,
                     num_segs: int = 1) -> torch.Tensor:

        if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are ["score", "prob", None]')

        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view((batch_size // num_segs, num_segs) + cls_scores.shape[1:])

        if self.average_clips is None:
            return cls_scores
        elif self.average_clips == 'prob':
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores


@MODELS.register_module()
class VideoClsHead(BaseVideoHead):
    def __init__(self,
                 in_channels,
                 num_classes,
                 spatial_type: str = 'avg',
                 consensus: dict = dict(type='AvgConsensus', dim=1),
                 dropout_ratio: float = 0.0,
                 is_shift: bool = True,
                 temporal_pool: bool = False,
                 init_cfg=None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio

        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        consensus_ = consensus.copy()
        consensus_type = consensus_.pop('type')
        if get_str_type(consensus_type) == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        normal_init(self.fc_cls, std=0.001)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if isinstance(x, tuple):
            x = x[-1]

        num_segs = x.shape[1]
        x = x.view((-1,) + x.shape[2:])

        if self.avg_pool is not None:
            x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)

        if self.is_shift and self.temporal_pool:
            cls_score = cls_score.view((-1, num_segs // 2) + cls_score.size()[1:])
        else:
            cls_score = cls_score.view((-1, num_segs) + cls_score.size()[1:])

        cls_score = self.consensus(cls_score)
        return cls_score.squeeze(1)


@MODELS.register_module()
class VideoClsHeadWithFrames(VideoClsHead):
    def __init__(self,
                 dropout_ratio_frame: float = 0.,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.fc_cls_frame = copy.deepcopy(self.fc_cls)

        if dropout_ratio_frame != 0:
            self.dropout_frame = nn.Dropout(p=dropout_ratio_frame)
        else:
            self.dropout_frame = None

        self.avg_pool_frame = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self) -> None:
        super().init_weights()
        normal_init(self.fc_cls_frame, std=0.001)

    def forward_frame(self, x: Tensor, **kwargs) -> Tensor:
        if isinstance(x, tuple):
            x = x[-1]

        if len(x.shape) == 5:
            x = x.flatten(0, 1)

        if self.avg_pool_frame is not None:
            x = self.avg_pool_frame(x)

        x = torch.flatten(x, 1)

        if self.dropout_frame is not None:
            x = self.dropout_frame(x)

        cls_scores = self.fc_cls_frame(x)
        return cls_scores.squeeze(1)
