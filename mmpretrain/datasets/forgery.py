import os.path as osp
from os import PathLike
from typing import List, Optional, Sequence, Union, Any

from mmengine.dataset import BaseDataset as _BaseDataset
from mmengine.fileio import join_path
from mmengine.utils import is_abs

from mmpretrain.registry import DATASETS, TRANSFORMS


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


@DATASETS.register_module()
class UnifiedImageDataset(_BaseDataset):
    def __init__(self,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 pipeline: Sequence = (),
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 ):
        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))

        ann_file = expanduser(ann_file)

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=transforms,
            test_mode=test_mode,
            lazy_init=lazy_init)

    def _join_prefix(self):
        if self.ann_file and not is_abs(self.ann_file) and '..' not in self.ann_file and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)

        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError(f'prefix should be a string, but got {type(prefix)}')
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f'raw_data_info: {raw_data_info} dose not contain prefix key'
                f'{prefix_key}, please check your data_prefix.')
            raw_data_info[prefix_key] = join_path(prefix, raw_data_info[prefix_key])

        return raw_data_info

    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()
        self._metainfo = dict()
        return data_list


@DATASETS.register_module()
class UnifiedVideoDataset(_BaseDataset):
    def __init__(self,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 pipeline: Sequence = (),
                 test_mode: bool = False,
                 lazy_init: bool = False,

                 start_index: int = 0,
                 modality: str = 'RGB',
                 ):
        if isinstance(data_prefix, str):
            data_prefix = dict(
                video_path=expanduser(data_prefix),
            )

        ann_file = expanduser(ann_file)

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        self.start_index = start_index
        self.modality = modality

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=transforms,
            test_mode=test_mode,
            lazy_init=lazy_init)

    def _join_prefix(self):
        if self.ann_file and not is_abs(self.ann_file) and '..' not in self.ann_file and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)

        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError(f'prefix should be a string, but got {type(prefix)}')
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f'raw_data_info: {raw_data_info} dose not contain prefix key'
                f'{prefix_key}, please check your data_prefix.')
            raw_data_info[prefix_key] = join_path(prefix, raw_data_info[prefix_key])

        return raw_data_info

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index
        return data_info

    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        data = self.pipeline(data_info)
        return data

    def load_data_list(self) -> List[dict]:
        data_list = super().load_data_list()
        self._metainfo = dict()
        return data_list
