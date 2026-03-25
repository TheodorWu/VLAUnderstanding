import einops
from eval.activation_writer import ActivationWriter
from eval.logger import Logger

class AttributionPatching():
    def __init__(self, config, model, tokenizer, perturbator, dataset, metric, device='cuda'):
        self.config = config
        self.model = model
        assert hasattr(self.model, "tracing_layers"), "Model must have attribute 'tracing_layers'."

        self.tokenizer = tokenizer # TODO: might move this to model later
        self.perturbator = perturbator
        self.dataset = dataset
        self.metric = metric
        self.device = device
        self.logger = Logger()
        self.writer = ActivationWriter(config)

        self.reset_patching_results()
        self.reset_collections()

    def reset_patching_results(self):
        # todo: might want to save this to disk instead of keeping in memory
        self.patching_results = []
        self.writer.reset_patching_results()

    def reset_collections(self):
        self.clean_out = []
        self.corrupted_out = []
        self.corrupted_grads = []

    def get_attribution_patching_data(self, batch):
        clean_batch = batch
        prompts = clean_batch["prompts"] # todo: will not work, have to check the key
        perturbed = clean_batch.copy()
        perturbed_prompts = self.perturbator.perturb(prompts).perturbed_prompts
        perturbed["prompts"] = perturbed_prompts

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

        with self.model.trace() as tracer:
        # Using nnsight's tracer.invoke context, we can batch the clean and the
        # corrupted runs into the same tracing context, allowing us to access
        # information generated within each of these runs within one forward pass

            with tracer.invoke(clean_batch) as invoker_clean:
                # Gather each layer's attention
                for layer_name in self.model.tracing_layers:
                    layer = getattr(self.model, layer_name)
                    # Get clean attention output for this layer
                    # across all attention heads
                    attn_out = layer.attn.c_proj.input
                    self.clean_out.append(attn_out.save())

            with tracer.invoke(corrupted_batch) as invoker_corrupted:
                # Gather each layer's attention and gradients
                for layer_name in self.model.tracing_layers:
                    layer = getattr(self.model, layer_name)
                    # Get corrupted attention output for this layer
                    # across all attention heads
                    attn_out = layer.attn.c_proj.input
                    self.corrupted_out.append(attn_out.save())
                    # save corrupted gradients for attribution patching
                    self.corrupted_grads.append(attn_out.grad.save())

                # Let's get the logits for the model's output
                # for the corrupted run
                logits = self.model.lm_head.output.save() # TODO: will not work, we are dealing with different model architectures

                # Our metric uses tensors saved on cpu, so we
                # need to move the logits to cpu.
                value = self.metric(logits.cpu())

                # We also need to run a backwards pass to
                # update gradient values
                value.backward()


    def attribution_patching_analysis(self):
        for corrupted_grad, corrupted, clean, layer in zip(
            self.corrupted_grads, self.corrupted_out, self.clean_out, range(len(self.clean_out))
        ):

            residual_attr = einops.reduce(
                corrupted_grad.value[:,-1,:] * (clean.value[:,-1,:] - corrupted.value[:,-1,:]),
                "batch (head dim) -> head",
                "sum",
                head = 12,
                dim = 64,
            )

            self.writer.add_patching_result(
                residual_attr.detach().cpu().numpy()
            )
