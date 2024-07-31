import re
from collections import OrderedDict

from mmengine.logging import print_log
from mmengine.model import PretrainedInit, update_init_info
from mmengine.registry import WEIGHT_INITIALIZERS


@WEIGHT_INITIALIZERS.register_module(name='PretrainedTSM')
class PretrainedInitTSM(PretrainedInit):
    def __init__(self, revise_keys=[(r'^module\.', '')], **kwargs):
        super().__init__(**kwargs)
        self.revise_keys = revise_keys  # NEW

    def __call__(self, module):
        from mmengine.runner.checkpoint import (_load_checkpoint_with_prefix,
                                                load_checkpoint,
                                                load_state_dict)
        if self.prefix is None:
            print_log(f'load model from: {self.checkpoint}', logger='current')
            load_checkpoint(
                module,
                self.checkpoint,
                map_location=self.map_location,
                strict=False,
                logger='current',
                revise_keys=self.revise_keys  # NEW
            )
        else:
            print_log(
                f'load {self.prefix} in model from: {self.checkpoint}',
                logger='current')
            state_dict = _load_checkpoint_with_prefix(
                self.prefix, self.checkpoint, map_location=self.map_location)

            # NEW
            for p, r in self.revise_keys:
                state_dict = OrderedDict(
                    {re.sub(p, r, k): v
                     for k, v in state_dict.items()})
            # END NEW

            load_state_dict(module, state_dict, strict=False, logger='current')

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())
