import copy

import torch
from mmengine.model import BaseModule, Sequential


class Sequential_TSM(Sequential):
    def __init__(self, *args, check=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.check = check

    def forward(self, x, **kwargs):
        for module in self:
            if not self.check or isinstance(module, (TemporalShiftModule, Sequential_TSM)):
                x = module(x, **kwargs)
            else:
                x = module(x)
        return x


class TemporalShiftModule(BaseModule):
    """Temporal shift module.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        module (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(self, module, num_segments=3, shift_div=8, is_share=True, is_shift=True):
        super().__init__()

        self.num_segments = num_segments
        self.shift_div = shift_div

        self.is_share = is_share
        self.is_shift = is_shift

        self.module = module
        if self.is_share:
            self.module_tsm = None
        else:
            self.module_tsm = copy.deepcopy(module)

    def get_module_tsm(self):
        if self.module_tsm is not None:
            return self.module_tsm
        return self.module

    def forward(self, x, with_temporal=True):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        if with_temporal and self.is_shift:
            # print('[called]')
            x = self.shift(x, self.num_segments, shift_div=self.shift_div)
            return self.get_module_tsm()(x)
        return self.module(x)

    @staticmethod
    def shift(x, num_segments, shift_div=3):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, C, H, W]
        n, c, h, w = x.size()

        # [N // num_segments, num_segments, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_segments, c, h * w)

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]

        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)

        # shift right on num_segments channel in `mid_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)

        # right_split: no shift

        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)

        # [N, C, H, W]
        # restore the original dimension
        return out.view(n, c, h, w)

    def copy_weights(self):
        # logger = MMLogger.get_current_instance()

        if self.module_tsm is not None:
            self.module_tsm.load_state_dict(self.module.state_dict())
            # logger.info(f'copy weights from {self.module} to {self.module_tsm}')
