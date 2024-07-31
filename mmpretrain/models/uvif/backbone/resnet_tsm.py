import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmengine.logging import MMLogger
from mmengine.registry import MODELS
from mmengine.runner.checkpoint import load_checkpoint

from mmpretrain.models.backbones.resnet import BasicBlock as _BasicBlock
from mmpretrain.models.backbones.resnet import Bottleneck as _Bottleneck
from mmpretrain.models.backbones.resnet import ResLayer as _ResLayer
from mmpretrain.models.backbones.resnet import ResNet
from .tsm import Sequential_TSM as Sequential
from .tsm import TemporalShiftModule


class BasicBlock(_BasicBlock):
    def forward(self, x, **kwargs):

        def _inner_forward(x, **kwargs):
            identity = x

            out = self.conv1(x, **kwargs)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x, **kwargs)
        else:
            out = _inner_forward(x, **kwargs)

        out = self.relu(out)

        return out


class Bottleneck(_Bottleneck):
    def forward(self, x, **kwargs):

        def _inner_forward(x, **kwargs):
            identity = x

            out = self.conv1(x, **kwargs)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x, **kwargs)
        else:
            out = _inner_forward(x, **kwargs)

        out = self.relu(out)

        return out


class ResLayer(_ResLayer):
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input


@MODELS.register_module()
class ResNetTSM(ResNet):
    """ResNet for TSM.

    Args:
        num_segments (int): Number of frame segments. Defaults to 8.
        is_shift (bool): Whether to make temporal shift in reset layers.
            Defaults to True.
        shift_div (int): Number of div for shift. Defaults to 8.
        shift_place (str): Places in resnet layers for shift, which is chosen
            from ['block', 'blockres'].
            If set to 'block', it will apply temporal shift to all child blocks
            in each resnet layer.
            If set to 'blockres', it will apply temporal shift to each `conv1`
            layer of all child blocks in each resnet layer.
            Defaults to 'blockres'.
        **kwargs (keyword arguments, optional): Arguments for ResNet.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_segments=16,
                 is_shift=True,
                 shift_div=8,
                 shift_place='blockres',
                 pretrained=None,
                 pretrained_3d=None,
                 is_share=True,
                 **kwargs):

        super().__init__(depth=depth, **kwargs)

        self.num_segments = num_segments
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place

        self.is_share = is_share

        self.pretrained = pretrained
        self.pretrained_3d = pretrained_3d

        self.build_tsm()

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def init_weights(self):
        logger = MMLogger.get_current_instance()

        if self.pretrained:
            load_checkpoint(
                self, self.pretrained, strict=False, logger=logger,
                revise_keys=[
                    ('.conv1.', '.conv1.module.'),
                ]
            )
        elif self.pretrained_3d:

            load_checkpoint(
                self, self.pretrained_3d, strict=False, logger=logger,
                revise_keys=[
                    ('backbone.', ''),
                    ('.net', '.module'),
                    ('conv1.conv', 'conv1'),
                    ('conv1.bn', 'bn1'),
                    ('conv2.conv', 'conv2'),
                    ('conv2.bn', 'bn2'),
                    ('conv3.conv', 'conv3'),
                    ('conv3.bn', 'bn3'),
                    ('downsample.conv', 'downsample.0'),
                    ('downsample.bn', 'downsample.1'),
                ]
            )
        else:
            super().init_weights()

        if not self.is_share:
            for i, (name, module) in enumerate(self.named_modules()):
                if isinstance(module, TemporalShiftModule):
                    module.copy_weights()
                    logger.info(f'Copy weights for {name}')

    def build_tsm(self):
        """Make temporal shift for some layers."""

        num_segment_list = [self.num_segments] * 4

        if num_segment_list[-1] <= 0:
            raise ValueError('num_segment_list[-1] must be positive')

        if self.shift_place == 'block':

            def make_block_temporal(stage, num_segments):
                """Make temporal shift on some blocks.

                Args:
                    stage (nn.Module): Model layers to be shifted.
                    num_segments (int): Number of frame segments.

                Returns:
                    nn.Module: The shifted blocks.
                """
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShiftModule(
                        b,
                        num_segments=num_segments,
                        shift_div=self.shift_div,
                        is_share=self.is_share,
                        is_shift=self.is_shift,
                    )
                return Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        elif 'blockres' in self.shift_place:
            n_round = 1

            # if len(list(self.layer3.children())) >= 23:
            #     n_round = 2

            def make_block_temporal(stage, num_segments):
                """Make temporal shift on some blocks.

                Args:
                    stage (nn.Module): Model layers to be shifted.
                    num_segments (int): Number of frame segments.

                Returns:
                    nn.Module: The shifted blocks.
                """
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShiftModule(
                            b.conv1,
                            num_segments=num_segments,
                            shift_div=self.shift_div,
                            is_share=self.is_share,
                            is_shift=self.is_shift,
                        )
                return Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])

        else:
            raise NotImplementedError

    def forward(self, x, **kwargs):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x, **kwargs)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
