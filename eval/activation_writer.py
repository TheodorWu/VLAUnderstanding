class ActivationWriter():
    def __init__(self, config):
        self.config = config
        self.reset_patching_results()
        self.chunk_size = config.get("chunk_size", 16)

    def __exit__(self, *args):
        if self.current_chunk_size > 0:
            self.save_chunk()

    def reset_patching_results(self):
        self.current_chunk_size = 0
        self.patching_results = []

    def add_patching_result(self, result):
        self.patching_results.append(result)
        self.current_chunk_size += 1
        if self.current_chunk_size >= self.chunk_size:
            self.save_chunk()
            self.reset_patching_results()

    def save_chunk(self):
        # Implementation for saving the current chunk of patching results
        pass
