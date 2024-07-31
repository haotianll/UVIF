from typing import Optional, Sequence, Union

from mmengine.hooks import LoggerHook as BaseLoggerHook

from mmpretrain.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class LoggerHook(BaseLoggerHook):
    def __init__(self, test_interval=100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.test_interval = test_interval

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        """Record logs after validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the validation
                loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
                Defaults to None.
            outputs (sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.test_interval):
            _, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'val')
            runner.logger.info(log_str)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """Record logs after testing iteration.

        Args:
            runner (Runner): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self.test_interval):
            _, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'test')
            runner.logger.info(log_str)
