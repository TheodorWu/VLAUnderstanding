import einops
from data.activation_writer import ActivationDataBatch, ActivationWriter
from eval.logger import Logger

class AttributionPatching():
    def __init__(self, config, model, tokenizer, perturbator, dataset, device='cuda'):
        self.config = config
        self.model = model
        assert hasattr(self.model, "tracing_layers"), "Model must have attribute 'tracing_layers'."
        self.tracing_layers = self.model.get_tracing_layers()

        self.tokenizer = tokenizer # TODO: might move this to model later
        self.perturbator = perturbator
        self.dataset = dataset
        self.device = device
        self.logger = Logger()
        self.writer = ActivationWriter(config)

        self.reset_collections()

    def get_tracing_target(self, name):
        paths_to_try = []
        if not name.startswith("model."):
            paths_to_try.append(f"model.{name}")
        paths_to_try.append(name)

        for path in paths_to_try:
            try:
                return self.model.get(path)
            except AttributeError:
                continue

        raise AttributeError(
            f"Could not resolve tracing target '{name}' or its variants."
        )

    def reset_collections(self):
        self.clean_out = {}
        self.corrupted_out = {}
        self.corrupted_grads = {}

    def get_attribution_patching_data(self, batch):
        clean_batch = batch
        prompts = clean_batch["task"]
        perturbed_prompts = self.perturbator.perturb({"prompt": prompts}).perturbed_prompts

        changed_indices = [i for i, (c, p) in enumerate(zip(prompts, perturbed_prompts)) if c != p]
        if not changed_indices:
            return None, None

        perturbed = clean_batch.copy()
        perturbed["task"] = perturbed_prompts

        clean_filtered = {k: [v[i] for i in changed_indices] for k, v in clean_batch.items()}
        perturbed_filtered = {k: [v[i] for i in changed_indices] for k, v in perturbed.items()}

        return clean_filtered, perturbed_filtered

    def main(self, unit_test=False):
        print("Starting attribution patching. Collecting activations and gradients for each batch in the dataset...")
        for batch in self.dataset:
            print("Processing next batch...")
            self.activation_tracing(batch)
            if unit_test:
                print("Unit test mode enabled; stopping after one batch.")
                break
        print("Attribution patching complete. All activations and gradients collected.")


    def activation_tracing(self, batch):
        print("Resetting stored activations and gradients...")
        self.reset_collections()
        print("Preparing clean and corrupted batches...")
        clean_batch, corrupted_batch = self.get_attribution_patching_data(batch)

        if clean_batch is None:
            return

        sample_ids = [ f"e{batch['episode_index'][i]}_i{batch['index'][i]}" for i in range(len(batch['episode_index']))]

        # Preprocess batches outside trace blocks - keeps tracing focused on model execution
        print("Preprocessing batches...")
        clean_batch_processed = self.model.preprocess_batch(clean_batch)
        corrupted_batch_processed = self.model.preprocess_batch(corrupted_batch)

        print("Tracing clean batch activations...")
        with self.model.trace() as tracer:
            with tracer.invoke(clean_batch_processed):
                for name in self.tracing_layers:
                     # test just one layer first
                    target = self.get_tracing_target(name)
                    self.clean_out[name] = target.input[0].save()

        print("Tracing corrupted batch activations and gradients...")
        with self.model.trace() as tracer:
            with tracer.invoke(corrupted_batch_processed):
                for name in self.tracing_layers:
                    target = self.get_tracing_target(name)
                    self.corrupted_out[name] = target.input[0].save()
                    target.input[0].retain_grad()

                loss = self.model.output
                loss.backward()

            with tracer.invoke():  # empty invoke to avoid execution-order conflicts
                for name in self.tracing_layers:
                    target = self.get_tracing_target(name)
                    self.corrupted_grads[name] = target.input[0].grad.save()

        # Log activations and gradients for each layer
        print("Writing traced data to the activation writer...")
        for name, clean in self.clean_out.items():
            self.writer.add_data(ActivationDataBatch(
                layer=name,
                sample_ids=sample_ids,
                clean=clean,
                corrupt=self.corrupted_out[name],
                gradients=self.corrupted_grads[name]
            ))
        print("Tracing complete.")
