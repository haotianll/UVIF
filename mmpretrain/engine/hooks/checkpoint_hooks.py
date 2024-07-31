from mmengine.hooks import CheckpointHook as BaseCheckpointHook

from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class CheckpointHook(BaseCheckpointHook):
    def before_train(self, runner) -> None:
        super().before_train(runner)

        self.filename_tmpl = self.filename_tmpl.replace('.pth', f'_{runner.timestamp}.pth')
