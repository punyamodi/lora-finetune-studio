from typing import Callable, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class ProgressCallback(TrainerCallback):
    def __init__(self, on_log: Optional[Callable] = None, on_epoch_end: Optional[Callable] = None):
        self.on_log_fn = on_log
        self.on_epoch_end_fn = on_epoch_end
        self.training_logs = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch, 3) if state.epoch else 0,
        }
        entry.update({k: round(v, 4) if isinstance(v, float) else v for k, v in logs.items()})
        self.training_logs.append(entry)
        if self.on_log_fn:
            self.on_log_fn(entry)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.on_epoch_end_fn:
            self.on_epoch_end_fn(state)

    def get_logs(self):
        return self.training_logs
