from typing import Dict, List, Optional, Sequence
from typing import Union

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from mmpretrain.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


@METRICS.register_module()
class UnifiedMetric(BaseMetric):
    """Metrics for MultiTask
    Args:
        task_metrics(dict): a dictionary in the keys are the names of the tasks
            and the values is a list of the metric corresponds to this task
    """

    default_prefix: Optional[str] = ''

    def __init__(self,
                 task_metrics: Dict,
                 collect_device: str = 'cpu',
                 print_copy_paste=False,

                 task_types={
                     'image': 'image',
                     'video': 'video',
                 },
                 ) -> None:

        self.task_metrics = task_metrics
        self.task_types = task_types

        super().__init__(collect_device=collect_device)

        self._metrics = {}
        for task_name in self.task_metrics.keys():
            self._metrics[task_name] = []
            for metric in self.task_metrics[task_name]:
                self._metrics[task_name].append(METRICS.build(metric))

        self.print_copy_paste = print_copy_paste

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for task_name in self.task_metrics.keys():
            if isinstance(data_samples, dict):
                _data_samples = data_samples[self.task_types[task_name]]
            else:
                _data_samples = data_samples

            sub_data_samples = []
            for data_sample in _data_samples:
                sub_data_sample = data_sample[task_name]
                sub_data_samples.append(sub_data_sample)
            for metric in self._metrics[task_name]:
                metric.process(data_batch, sub_data_samples)

    def compute_metrics(self, results: list) -> dict:
        raise NotImplementedError('compute metrics should not be used here directly')

    def evaluate(self, size):
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.
        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are
            "{task_name}_{metric_name}" , and the values
            are corresponding results.
        """
        metrics = {}
        for task_name in self._metrics:
            for metric in self._metrics[task_name]:
                name = metric.__class__.__name__
                if name == 'UnifiedMetric' or metric.results:
                    results = metric.evaluate(size)
                else:
                    results = {metric.__class__.__name__: 0}
                for key in results:
                    name = f'{task_name}/{key}'
                    if name in results:
                        raise ValueError(f'There are multiple metric results with the same metric name {name}.')
                    metrics[name] = results[key]

        if self.print_copy_paste:
            copy_paste = []
            for key, value in metrics.items():
                copy_paste.append('{:.4f}'.format(value))
            metrics['copy_paste'] = ' '.join(copy_paste)

        return metrics


@METRICS.register_module()
class BalancedAccuracy(BaseMetric):
    r"""Accuracy Balanced over classes, introduced in ForgeryNet.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = ''

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()

            result['pred_label'] = data_sample['pred_label'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        pred = torch.cat([res['pred_label'] for res in results])

        metrics = self.calculate(pred, target)

        return metrics

    @staticmethod
    def calculate(
            pred: Union[torch.Tensor, np.ndarray, Sequence],
            target: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the accuracy.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            thrs (Sequence[float | None]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. None means no thresholds.
                Defaults to (0., ).
            thrs (Sequence[float]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. Defaults to (0., ).

        Returns:
            torch.Tensor | List[List[torch.Tensor]]: Accuracy.

            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only return a top-1 accuracy
              tensor, and ignore the argument ``topk` and ``thrs``.
            - List[List[torch.Tensor]]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the accuracy on each ``topk``
              and ``thrs``. And the first dim is ``topk``, the second dim is
              ``thrs``.
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)

        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match " \
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            pred = pred.tolist()
            target = target.tolist()
        else:
            pred = pred.tolist()
            target = target.flatten().tolist()

        mAcc = balanced_accuracy_score(target, pred)
        metrics = {
            'mAcc': 100. * mAcc,
        }
        return metrics


@METRICS.register_module()
class AUC(BaseMetric):
    r"""AUC
    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = ''

    def __init__(self,
                 neg_index=0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.neg_index = neg_index

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()

            result['pred_score'] = data_sample['pred_score'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()

            result['pred_score'] = 1 - result['pred_score'][self.neg_index]
            result['gt_label'] = torch.tensor([0]) if result['gt_label'] == self.neg_index else torch.tensor([1])

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.
        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.

        target = torch.cat([res['gt_label'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])

        metrics = self.calculate(pred, target)
        return metrics

    def calculate(
            self,
            pred: Union[torch.Tensor, np.ndarray, Sequence],
            target: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> Dict:
        """Calculate the precision, recall, f1-score and support.
        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
        Returns:
            - float: 100. * AUC
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match " \
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            pred_score = pred.tolist()
            target = target.tolist()
        else:
            pred_score = pred.tolist()
            target = target.flatten().tolist()

        metrics = dict()

        auc = roc_auc_score(target, pred_score)
        metrics['AUC'] = 100. * auc

        return metrics
