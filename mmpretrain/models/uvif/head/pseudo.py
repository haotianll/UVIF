from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class VideoPseudoLabeling(BaseModule):
    def __init__(self,
                 loss_pseudo_module: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(loss_pseudo_module, nn.Module):
            loss_pseudo_module = MODELS.build(loss_pseudo_module)
        self.loss_pseudo_module = loss_pseudo_module

    def loss(self, inputs: torch.Tensor, data_samples: Optional[list] = None, decoder_modules=None, **kwargs):
        feats = inputs['video']
        feats_weak = inputs['video_weak']

        frame_logits = decoder_modules['video'].forward_frame(feats)

        losses = dict()

        with torch.no_grad():
            feats_weak = tuple([x.view((-1,) + x.shape[2:]) for x in feats_weak])
            pseudo_labels = decoder_modules['image'](feats_weak)
            pseudo_labels = torch.softmax(pseudo_labels, dim=1)

        losses['loss_pseudo'] = self.loss_pseudo_module(frame_logits, pseudo_labels.detach(), **kwargs)
        return losses
