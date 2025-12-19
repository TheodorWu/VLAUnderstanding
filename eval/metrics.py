class Metrics:
    def compute_accuracy(self, predictions, targets):
        correct = sum(p == t for p, t in zip(predictions, targets))
        return correct / len(targets) if targets else 0
