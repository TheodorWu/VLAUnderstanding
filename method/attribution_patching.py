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
        # First, try direct lookup with model prefix (common case)
        paths_to_try = []
        if not name.startswith("model."):
            paths_to_try.append(f"model.{name}")
        paths_to_try.append(name)

        for path in paths_to_try:
            try:
                proxy = self.model.get(path)
            except AttributeError:
                pass

            return proxy.o_proj

        # If both fail, raise a clear error
        raise AttributeError(
            f"Could not resolve tracing target '{name}' or its variants. "
            f"Available model: {self.model}"
        )

    def reset_collections(self):
        self.clean_out = {}
        self.corrupted_out = {}
        self.corrupted_grads = {}

    def get_attribution_patching_data(self, batch):
        clean_batch = batch
        prompts = clean_batch["task"] # todo: will not work, have to check the key
        perturbed = clean_batch.copy()
        perturbed_prompts = self.perturbator.perturb({"prompt": prompts}).perturbed_prompts
        perturbed["task"] = perturbed_prompts

        return clean_batch, perturbed

    def main(self, unit_test=False):
        for batch in self.dataset:
            self.activation_tracing(batch)
            self.attribution_patching_analysis()
            if unit_test:
                break

    def activation_tracing(self, batch):
        self.reset_collections()
        clean_batch, corrupted_batch = self.get_attribution_patching_data(batch)
        sample_ids = [ f"e{batch['episode_index'][i]}_i{batch['index'][i]}" for i in range(len(batch['episode_index']))]

        # todo: write batch to disk using activation writer

        # Preprocess batches outside trace blocks - keeps tracing focused on model execution
        clean_batch_processed = self.model.preprocess_batch(clean_batch)
        corrupted_batch_processed = self.model.preprocess_batch(corrupted_batch)

        with self.model.trace() as tracer:
            with tracer.invoke(clean_batch_processed):
                for name in self.tracing_layers:
                     # test just one layer first
                    target = self.get_tracing_target(name)
                    self.clean_out[name] = target.input[0].save()

        with self.model.trace() as tracer:
            with tracer.invoke(corrupted_batch_processed):
                for name in self.tracing_layers:
                    target = self.get_tracing_target(name)
                    self.corrupted_out[name] = target.input[0].save()
                    self.corrupted_grads[name] = target.input[0].grad.save()

            output = self.model.get_output_proxy()
            value = self.model.metric(output)
            value.backward()

        # Log activations and gradients for each layer
        for name in self.clean_out.keys():
            self.writer.add_data(ActivationDataBatch(
                layer=name,
                sample_ids=sample_ids,
                activations=self.clean_out[name].value,   # clean acts, no grads needed
            ))
            self.writer.add_data(ActivationDataBatch(
                layer=name,
                sample_ids=sample_ids,
                activations=self.corrupted_out[name].value,
                gradients=self.corrupted_grads[name].value,
            ))

    def attribution_patching_analysis(self):
        for layer in self.clean_out.keys():
            corrupted_grad = self.corrupted_grads[layer]
            corrupted = self.corrupted_out[layer]
            clean = self.clean_out[layer]

            residual_attr = einops.reduce(
                corrupted_grad.value[:,-1,:] * (clean.value[:,-1,:] - corrupted.value[:,-1,:]),
                "batch (head dim) -> head",
                "sum",
                head = 12,
                dim = 64,
            )

            self.writer.add_data(
                residual_attr.detach().cpu().numpy()
            )
