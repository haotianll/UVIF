from typing import Optional, Sequence

import mmengine
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from mmcv.transforms import BaseTransform

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures.unified_data_sample import UnifiedDataSample
from .formatting import to_tensor


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


# REF: mmcv.transforms import LoadImageFromFile
@TRANSFORMS.register_module()
class FaceCrop(BaseTransform):
    def __init__(self,
                 scale=1., ):
        self.scale = scale

    @staticmethod
    def rescale_bbox(bbox, height, width, scale):
        x1, y1, x2, y2 = bbox

        w = int((x2 - x1) * scale)
        h = int((y2 - y1) * scale)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        x1 = max(int(center_x - w // 2), 0)
        y1 = max(int(center_y - h // 2), 0)
        w = min(width - x1, w)
        h = min(height - y1, h)

        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def transform(self, results: dict) -> Optional[dict]:

        face_data = results.get('face_data', None)
        if len(face_data) > 0:
            bbox = face_data
        else:
            bbox = None

        results['full_shape'] = results['ori_shape']

        if bbox is not None:
            h, w = results['ori_shape']

            bbox = self.rescale_bbox(bbox, h, w, scale=self.scale)
            x1, y1, x2, y2 = bbox

            img = results['img']
            img = img[y1:y2, x1:x2]

            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]

            results['face_pad_info'] = x1, w - x2, y1, h - y2  # top, bottom, left, right

            if results.get('gt_seg_map') is not None:
                gt_seg_map = results['gt_seg_map']
                gt_seg_map = gt_seg_map[y1:y2, x1:x2]
                results['gt_seg_map'] = gt_seg_map

        return results


# REF: PackInputs
@TRANSFORMS.register_module()
class PackImageInputs(BaseTransform):
    DEFAULT_META_KEYS = (
        'sample_idx', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction',
        'gt_seg_map', 'seg_map_path', 'face_pad_info', 'full_shape',
        'face_data'
    )

    def __init__(self,
                 input_key='img',
                 algorithm_keys=(),
                 meta_keys=DEFAULT_META_KEYS,
                 ):
        self.input_key = input_key
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackImageInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            if input_.ndim == 2:  # For grayscale image.
                input_ = np.expand_dims(input_, -1)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_.transpose(2, 0, 1))
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                # convert to tensor first to accelerate, see
                # https://github.com/open-mmlab/mmdetection/pull/9533
                input_ = to_tensor(input_).permute(2, 0, 1).contiguous()
            else:
                # convert input with other shape to tensor without permute,
                # like video input (num_crops, C, T, H, W).
                input_ = to_tensor(input_)
        elif isinstance(input_, Image.Image):
            input_ = F.pil_to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f'Unsupported input type {type(input_)}.')

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        data_sample = UnifiedDataSample(data_type='image')

        # Set default keys
        if 'gt_label' in results:
            data_sample.tasks('image').set_gt_label(results['gt_label'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results
