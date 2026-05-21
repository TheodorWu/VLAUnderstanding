import wandb
class Logger:
    def log_metric(self, name, value, step):
        print(f"Step {step}: {name} = {value}")
        wandb.log({name: value}, step=step)
