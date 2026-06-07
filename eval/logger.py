import wandb
import numpy as np

class Logger:
    def __init__(self):
        self.accumulated_metrics = {}

    def _transform_to_numpy(self, value):
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return value

    def log_metric(self, name, value, step):
        value = self._transform_to_numpy(value)
        print(f"Step {step}: {name} = {value}")
        if wandb.run is not None:
            wandb.log({name: value}, step=step)
        if not name in self.accumulated_metrics:
            self.accumulated_metrics[name] = []
        self.accumulated_metrics[name].append(value)

    def __exit__(self, exc_type, exc, tb):
        print("Final accumulated metrics:")
        for name, values in self.accumulated_metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            if wandb.run is not None:
                wandb.log({f"{name}_mean": mean, f"{name}_std": std})
            print(f"{name}: mean = {mean}, std = {std}")
