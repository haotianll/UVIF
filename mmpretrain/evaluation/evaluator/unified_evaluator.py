from typing import Any, Optional, Sequence

from mmengine.evaluator import Evaluator
from mmengine.structures import BaseDataElement

from mmpretrain.registry import EVALUATORS


@EVALUATORS.register_module()
class UnifiedEvaluator(Evaluator):
    def process(self,
                data_samples: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model,
                and the ground truth of the validation set.
            data_batch (Any, optional): A batch of data from the dataloader.
        """

        if isinstance(data_samples, dict):
            _data_samples = dict()
            for key, samples in data_samples.items():
                _data_samples[key] = []
                for data_sample in samples:
                    if isinstance(data_sample, BaseDataElement):
                        _data_samples[key].append(data_sample.to_dict())
                    else:
                        _data_samples[key].append(data_sample)
        else:
            _data_samples = []
            for data_sample in data_samples:
                if isinstance(data_sample, BaseDataElement):
                    _data_samples.append(data_sample.to_dict())
                else:
                    _data_samples.append(data_sample)

        for metric in self.metrics:
            metric.process(data_batch, _data_samples)
