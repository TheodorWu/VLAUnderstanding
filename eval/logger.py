class Logger:
    def log_metric(self, name, value, step):
        print(f"Step {step}: {name} = {value}")
