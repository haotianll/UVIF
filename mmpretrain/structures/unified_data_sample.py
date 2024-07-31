from typing import Sequence, Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement
from mmengine.utils import is_str

LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence]


def format_label(value: LABEL_TYPE) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The foramtted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def format_score(value: SCORE_TYPE) -> torch.Tensor:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence): Score values.

    Returns:
        :obj:`torch.Tensor`: The foramtted score tensor.
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


# REF: DataSample (mmpretrain)
class ClsDataSample(BaseDataElement):
    """A general data structure interface.

    It's used as the interface between different components.

    The following fields are convention names in MMPretrain, and we will set or
    get these fields in data transforms, models, and metrics if needed. You can
    also set any new fields for your need.

    Meta fields:
        img_shape (Tuple): The shape of the corresponding input image.
        ori_shape (Tuple): The original shape of the corresponding image.
        sample_idx (int): The index of the sample in the dataset.
        num_classes (int): The number of all categories.

    Data fields:
        gt_label (tensor): The ground truth label.
        gt_score (tensor): The ground truth score.
        pred_label (tensor): The predicted label.
        pred_score (tensor): The predicted score.

    """

    def set_gt_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``gt_label``."""
        self.set_field(format_label(value), 'gt_label', dtype=torch.Tensor)
        return self

    def set_gt_score(self, value: SCORE_TYPE) -> 'DataSample':
        """Set ``gt_score``."""
        score = format_score(value)
        self.set_field(score, 'gt_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be ' \
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    def set_pred_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``pred_label``."""
        self.set_field(format_label(value), 'pred_label', dtype=torch.Tensor)
        return self

    def set_pred_score(self, value: SCORE_TYPE):
        """Set ``pred_label``."""
        score = format_score(value)
        self.set_field(score, 'pred_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be ' \
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    def __repr__(self) -> str:
        """Represent the object."""

        def dump_items(items, prefix=''):
            return '\n'.join(f'{prefix}{k}: {v}' for k, v in items)

        repr_ = ''
        if len(self._metainfo_fields) > 0:
            repr_ += '\n\nMETA INFORMATION\n'
            repr_ += dump_items(self.metainfo_items(), prefix=' ' * 4)
        if len(self._data_fields) > 0:
            repr_ += '\n\nDATA FIELDS\n'
            repr_ += dump_items(self.items(), prefix=' ' * 4)

        repr_ = f'<{self.__class__.__name__}({repr_}\n\n) at {hex(id(self))}>'
        return repr_


class UnifiedDataSample(BaseDataElement):
    def __init__(self,
                 data_type='image',
                 tasks=['video', ],
                 metainfo=None, **kwargs):
        super().__init__(metainfo=metainfo, **kwargs)

        self.data_type = data_type

        if data_type == 'image':
            self._tasks = tasks
            self.set_field(ClsDataSample(), 'image')
        elif data_type == 'video':
            self._tasks = tasks
            self.set_field(ClsDataSample(), 'video')
        else:
            raise NotImplementedError(data_type)

    def tasks(self, task_name):
        return getattr(self, task_name)
