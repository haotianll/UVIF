import copy
import inspect
import io
import math
import os.path as osp
from typing import Callable, Union, Dict, List, Optional, Tuple, Sequence

import albumentations
import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.processing import Resize, RandomResize, RandomFlip
from mmengine.fileio import FileClient

from mmpretrain.datasets.transforms.image import PackImageInputs, to_tensor
from mmpretrain.datasets.transforms.processing import RandomCrop
from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures.unified_data_sample import UnifiedDataSample

Transform = Union[dict, Callable[[dict], dict]]


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


@TRANSFORMS.register_module()
class VideoDecordInit(BaseTransform):
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - filename

    Added Keys:

        - video_reader
        - total_frames
        - fps

    Args:
        io_backend (str): io backend where frames are store.
            Defaults to ``'disk'``.
        num_threads (int): Number of thread to decode the video. Defaults to 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend: str = 'disk',
                 num_threads: int = 1,
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def _get_video_reader(self, filename: str) -> object:
        if osp.splitext(filename)[0] == filename:
            filename = filename + '.mp4'
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        file_obj = io.BytesIO(self.file_client.get(filename))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        return container

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord initialization.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        container = self._get_video_reader(results['video_path'])
        results['total_frames'] = len(container)

        results['video_reader'] = container
        results['avg_fps'] = container.get_avg_fps()
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@TRANSFORMS.register_module()
class VideoSampleFrames(BaseTransform):
    """Sample frames from the video.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Defaults to 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Defaults to False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Defaults to False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Defaults to 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Defaults to False.
        target_fps (optional, int): Convert input videos with arbitrary frame
            rates to the unified target FPS before sampling frames. If
            ``None``, the frame rate will not be adjusted. Defaults to
            ``None``.
    """

    def __init__(self,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 test_mode: bool = False,
                 keep_tail_frames: bool = False,
                 target_fps: Optional[int] = None,
                 **kwargs) -> None:

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.target_fps = target_fps
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames: int,
                         ori_clip_len: float) -> np.array:
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int,
                        ori_clip_len: float) -> np.array:
        """Get clip offsets in test mode.

        If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        if self.clip_len == 1:  # 2D recognizer
            # assert self.frame_interval == 1
            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:  # 3D recognizer
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1
                # align test sample strategy with `PySlowFast` repo
                if self.target_fps is not None:
                    offset_between = np.floor(max_offset / float(num_segments))
                    clip_offsets = np.arange(num_clips) * offset_between
                else:
                    offset_between = max_offset / float(num_segments)
                    clip_offsets = np.arange(num_clips) * offset_between
                    clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)

        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio: float) -> float:
        """calculate length of clip segment for different strategy.

        Args:
            fps_scale_ratio (float): Scale ratio to adjust fps.
        """
        if self.target_fps is not None:
            # align test sample strategy with `PySlowFast` repo
            ori_clip_len = self.clip_len * self.frame_interval
            ori_clip_len = np.maximum(1, ori_clip_len * fps_scale_ratio)
        elif self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval

        return ori_clip_len

    def transform(self, results: dict) -> dict:
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        # if can't get fps, same value of `fps` and `target_fps`
        # will perform nothing
        fps = results.get('avg_fps')
        if self.target_fps is None or not fps:
            fps_scale_ratio = 1.0
        else:
            fps_scale_ratio = fps / self.target_fps
        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets = self._sample_clips(total_frames, ori_clip_len)

        if self.target_fps:
            frame_inds = clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len).astype(np.int32)
        else:
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@TRANSFORMS.register_module()
class VideoDecordDecode(BaseTransform):
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required Keys:

        - video_reader
        - frame_inds

    Added Keys:

        - imgs
        - original_shape
        - img_shape

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets.
            Defaults to ``'accurate'``.
    """

    def __init__(self, mode: str = 'accurate') -> None:
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def _decord_load_frames(self, container: object,
                            frame_inds: np.ndarray) -> List[np.ndarray]:
        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy())
        return imgs

    def transform(self, results: Dict) -> Dict:
        """Perform the Decord decoding.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = self._decord_load_frames(container, frame_inds)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str


@TRANSFORMS.register_module()
class VideoFaceCrop(BaseTransform):
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

    def transform(self, results):
        if len(results['face_data']) > 0:
            results['face_data'] = [results['face_data'][i] for i in results['frame_inds']]

            for i, bbox in enumerate(results['face_data']):
                h, w = results['imgs'][i].shape[:2]
                bbox = self.rescale_bbox(bbox, h, w, self.scale)
                x1, y1, x2, y2 = bbox
                results['imgs'][i] = results['imgs'][i][y1:y2, x1:x2]
        return results


# NEW: based on mmcv
@TRANSFORMS.register_module()
class VideoResize(Resize):
    def mmcv_resize_img(self, img, scale) -> None:
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                img,
                scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend=self.backend)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = img.shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img,
                scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend=self.backend)
        return img

    def _resize_imgs(self, results: dict) -> None:
        results['imgs'] = [
            self.mmcv_resize_img(img, results['scale']) for img in results['imgs']
        ]
        results['img_shape'] = results['imgs'][0].shape[:2]
        results['keep_ratio'] = self.keep_ratio

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['imgs'][0].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)  # type: ignore
            raise NotImplementedError('TODO')

        self._resize_imgs(results)
        return results


# NEW: based on RandomResize
@TRANSFORMS.register_module()
class VideoRandomResize(RandomResize):
    """Random resize images & bbox & keypoints.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if ``scale`` is a sequence of tuple

    .. math::
        target\\_scale[0] \\sim Uniform([scale[0][0], scale[1][0]])
    .. math::
        target\\_scale[1] \\sim Uniform([scale[0][1], scale[1][1]])

    Following the resize order of weight and height in cv2, ``scale[i][0]``
    is for width, and ``scale[i][1]`` is for height.

    - if ``scale`` is a tuple

    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[0]
    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[1]

    Following the resize order of weight and height in cv2, ``ratio_range[0]``
    is for width, and ``ratio_range[1]`` is for height.

    - if ``keep_ratio`` is True, the minimum value of ``target_scale`` will be
      used to set the shorter side and the maximum value will be used to
      set the longer side.

    - if ``keep_ratio`` is False, the value of ``target_scale`` will be used to
      reisze the width and height accordingly.

    Required Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (tuple or Sequence[tuple]): Images scales for resizing.
            Defaults to None.
        ratio_range (tuple[float], optional): (min_ratio, max_ratio).
            Defaults to None.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
            self,
            scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
            ratio_range: Tuple[float, float] = None,
            resize_type: str = 'VideoResize',
            **resize_kwargs,
    ) -> None:
        self.scale = scale
        self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})


# NEW: based on RandomCrop
@TRANSFORMS.register_module()
class VideoRandomCrop(RandomCrop):
    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        if self.padding is not None:
            results['imgs'] = [
                mmcv.impad(img, padding=self.padding, pad_val=self.pad_val) for img in results['imgs']
            ]

        # pad img if needed
        if self.pad_if_needed:
            h_pad = math.ceil(max(0, self.crop_size[0] - results['imgs'][0].shape[0]) / 2)
            w_pad = math.ceil(max(0, self.crop_size[1] - results['imgs'][0].shape[1]) / 2)

            results['imgs'] = [
                mmcv.impad(
                    img,
                    padding=(w_pad, h_pad, w_pad, h_pad),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode
                )
                for img in results['imgs']
            ]

        offset_h, offset_w, target_h, target_w = self.rand_crop_params(results['imgs'][0])
        results['imgs'] = [
            mmcv.imcrop(
                img,
                np.array([offset_w, offset_h, offset_w + target_w - 1, offset_h + target_h - 1, ])
            )
            for img in results['imgs']
        ]
        results['img_shape'] = results['imgs'][0].shape
        return results


# NEW: based on mmcv
@TRANSFORMS.register_module()
class VideoRandomFlip(RandomFlip):
    def _flip(self, results: dict) -> None:
        results['imgs'] = [
            mmcv.imflip(img, direction=results['flip_direction']) for img in results['imgs']
        ]


class VideoGaussNoise(albumentations.GaussNoise):
    def apply(self, img, gauss=None, **params):
        from albumentations import functional as F
        if img.shape != gauss.shape:
            h, w = img.shape[:2]
            gauss = cv2.resize(gauss, (w, h), interpolation=cv2.INTER_LINEAR)
        return F.gauss_noise(img, gauss=gauss)


@TRANSFORMS.register_module(['VideoAlbumentations', 'VideoAlbu'])
class VideoAlbumentations(BaseTransform):

    def __init__(self, transforms: List[Dict], keymap: Optional[Dict] = None):
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')
        else:
            from albumentations import Compose as albu_Compose

        assert isinstance(transforms, list), 'transforms must be a list.'
        if keymap is not None:
            assert isinstance(keymap, dict), 'keymap must be None or a dict. '

        self.transforms = transforms

        self.aug = albu_Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = dict(img='image')
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: Dict):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg, 'each item in ' \
                                                        "transforms must be a dict with keyword 'type'."
        args = cfg.copy()

        obj_type = args.pop('type')
        if obj_type == 'GaussNoise':
            obj_cls = VideoGaussNoise
        elif mmengine.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def transform(self, results: Dict) -> Dict:
        """Transform function to perform albumentations transforms.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results, 'img' and 'img_shape' keys are
                updated in result dict.
        """
        assert 'imgs' in results, 'No `img` field in the input.'

        # results = self.mapper(results, self.keymap_to_albu)
        # results = self.aug(**results)
        # results = self.mapper(results, self.keymap_back)

        inputs = {'image': results['imgs'][0]}
        for i, img in enumerate(results['imgs'][1:]):
            inputs[f'image{i + 1}'] = img

        results['imgs'] = list(self.aug(**inputs).values())

        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={repr(self.transforms)})'
        return repr_str


# REF: FormatShape (mmaction2), support MultiView inputs
@TRANSFORMS.register_module()
class VideoFormatShape(BaseTransform):
    """Format final imgs shape to the given input_format.

    Required keys:
        - imgs (optional)
        - heatmap_imgs (optional)
        - num_clips
        - clip_len

    Modified Keys:
        - imgs (optional)
        - input_shape (optional)

    Added Keys:
        - heatmap_input_shape (optional)

    Args:
        input_format (str): Define the final data format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, input_format: str, collapse: bool = False) -> None:
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in [
            'NCTHW', 'NCHW', 'NCHW_Flow', 'NCTHW_Heatmap', 'NPTCHW'
        ]:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform_results(self, results: Dict) -> Dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])

        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * T
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            if 'imgs' in results:
                imgs = results['imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1,) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

            if 'heatmap_imgs' in results:
                imgs = results['heatmap_imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                # clip_len must be a dict
                clip_len = clip_len['Pose']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1,) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['heatmap_imgs'] = imgs
                results['heatmap_input_shape'] = imgs.shape

        elif self.input_format == 'NCTHW_Heatmap':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x C x H x W
            imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
            # N_crops x N_clips x C x T x H x W
            imgs = imgs.reshape((-1,) + imgs.shape[2:])
            # M' x C x T x H x W
            # M' = N_crops x N_clips
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW':
            imgs = results['imgs']
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW_Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            imgs = np.stack([x_flow, y_flow], axis=-1)

            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x T x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = T x C
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x T
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        if self.collapse:
            assert results['imgs'].shape[0] == 1
            results['imgs'] = results['imgs'].squeeze(0)
            results['input_shape'] = results['imgs'].shape

        return results

    def transform(self, results: Dict) -> Dict:
        if not isinstance(results['imgs'], List) or not isinstance(results['imgs'][0], List):
            return self.transform_results(results)

        imgs_list = results.pop('imgs')
        out_results = copy.deepcopy(results)
        out_results['imgs'] = []

        for i, imgs in enumerate(imgs_list):
            result = copy.deepcopy(results)
            result['imgs'] = imgs

            result = self.transform_results(result)
            out_results['imgs'].append(result['imgs'])

            if i == 0:
                result.pop('imgs')
                out_results.update(result)

        return out_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


# REF: wrappers.MultiView
@TRANSFORMS.register_module()
class VideoMultiView(BaseTransform):
    """A transform wrapper for multiple views of an image.

    Args:
        transforms (list[dict | callable], optional): Sequence of transform
            object or config dict to be wrapped.
        mapping (dict): A dict that defines the input key mapping.
            The keys corresponds to the inner key (i.e., kwargs of the
            ``transform`` method), and should be string type. The values
            corresponds to the outer keys (i.e., the keys of the
            data/results), and should have a type of string, list or dict.
            None means not applying input mapping. Default: None.
        allow_nonexist_keys (bool): If False, the outer keys in the mapping
            must exist in the input data, or an exception will be raised.
            Default: False.

    Examples:
        >>> # Example 1: MultiViews 1 pipeline with 2 views
        >>> pipeline = [
        >>>     dict(type='MultiView',
        >>>         num_views=2,
        >>>         transforms=[
        >>>             [
        >>>                dict(type='Resize', scale=224))],
        >>>         ])
        >>> ]
        >>> # Example 2: MultiViews 2 pipelines, the first with 2 views,
        >>> # the second with 6 views
        >>> pipeline = [
        >>>     dict(type='MultiView',
        >>>         num_views=[2, 6],
        >>>         transforms=[
        >>>             [
        >>>                dict(type='Resize', scale=224)],
        >>>             [
        >>>                dict(type='Resize', scale=224),
        >>>                dict(type='RandomSolarize')],
        >>>         ])
        >>> ]
    """

    def __init__(self,
                 transforms: List[List[Transform]],
                 num_views: Union[int, List[int]], ) -> None:

        if isinstance(num_views, int):
            num_views = [num_views]
        assert isinstance(num_views, List)
        assert len(num_views) == len(transforms)
        self.num_views = num_views

        self.pipelines = []
        for trans in transforms:
            pipeline = Compose(trans)
            self.pipelines.append(pipeline)

        self.transforms = []
        for i in range(len(num_views)):
            self.transforms.extend([self.pipelines[i]] * num_views[i])

    def transform(self, results: dict) -> dict:
        """Apply transformation to inputs.

        Args:
            results (dict): Result dict from previous pipelines.

        Returns:
            dict: Transformed results.
        """
        multi_views_outputs = dict(imgs=[])
        for trans in self.transforms:
            inputs = copy.deepcopy(results)
            outputs = trans(inputs)

            multi_views_outputs['imgs'].append(outputs['imgs'])
        results.update(multi_views_outputs)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + '('
        for i, p in enumerate(self.pipelines):
            repr_str += f'\nPipeline {i + 1} with {self.num_views[i]} views:\n'
            repr_str += str(p)
        repr_str += ')'
        return repr_str


@TRANSFORMS.register_module()
class VideoIdentity(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        return results


# REF: PackActionInputs (mmaction2) + PackInputs (mmpretrain)
@TRANSFORMS.register_module()
class PackVideoInputs(BaseTransform):
    DEFAULT_META_KEYS = (
        'sample_idx', 'video_path',
        'img_shape', 'img_key', 'video_id', 'timestamp',
        'face_data'
    )

    def __init__(self,
                 input_key='imgs',
                 collect_keys=[],
                 algorithm_keys=(),
                 meta_keys=DEFAULT_META_KEYS,
                 ):
        self.input_key = input_key
        self.collect_keys = collect_keys
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackImageInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            input_ = to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f'Unsupported input type {type(input_)}.')
        return input_

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        if self.collect_keys is not None:
            for key in self.collect_keys:
                packed_results['inputs'][key] = to_tensor(results[key])

        data_sample = UnifiedDataSample(
            data_type='video',
        )

        # Set default keys
        if 'gt_label' in results:
            data_sample.tasks('video').set_gt_label(results['gt_label'])

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
