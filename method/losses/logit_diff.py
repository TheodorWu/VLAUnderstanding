class LogitDifference:
    def __call__(self, logits, answer_token_indices):
        # TODO: this might assume a wrong shape. Also I have to decide what to do with the answer tokens.
        logits = logits[:, -1, :]
        correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
        incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
        return (correct_logits - incorrect_logits).mean()
