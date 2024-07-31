from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmpretrain.models.necks.gap import GlobalAveragePooling
from mmpretrain.registry import MODELS


class BaseImageHead(BaseModule):
    def __init__(self,
                 loss_module: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        if not isinstance(loss_module, nn.Module):
            loss_module = MODELS.build(loss_module)
        self.loss_module = loss_module

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

        # compute loss
        losses = dict()
        loss = self.loss_module(cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses[f'loss_{task_name}'] = loss
        return losses

    def predict(self, pred, data_samples, task_name=None):
        """Post-process the output of head."""

        # Old implementation for UnifiedHead
        if isinstance(pred, dict) and task_name in pred.keys():
            pred = pred[task_name]

        pred_scores = F.softmax(pred, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        for data_sample, score, label in zip(data_samples, pred_scores, pred_labels):
            sub_data_sample = data_sample.tasks(task_name)
            sub_data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples


@MODELS.register_module()
class ImageClsHead(BaseImageHead):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 dropout_ratio: float = 0.,
                 init_cfg: Optional[dict] = dict(type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(f'num_classes={num_classes} must be a positive integer')

        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.gap = GlobalAveragePooling()
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        return self.gap(feats[-1])

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)

        if self.dropout is not None:
            pre_logits = self.dropout(pre_logits)

        cls_score = self.fc(pre_logits)
        return cls_score
