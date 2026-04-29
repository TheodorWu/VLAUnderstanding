import einops
from data.activation_writer import ActivationDataBatch, ActivationWriter
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

        self.reset_collections()

    def reset_collections(self):
        self.clean_out = {}
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
        sample_ids = batch.sample_ids

        # todo: write batch to disk using activation writer

        with self.model.trace() as tracer:
        # Using nnsight's tracer.invoke context, we can batch the clean and the
        # corrupted runs into the same tracing context, allowing us to access
        # information generated within each of these runs within one forward pass
            with tracer.invoke(clean_batch):
                # Gather each layer's attention
                for name, layer in self.model.tracing_layers:
                    target = self.model.get_tracing_target(layer)
                    self.clean_out[name] = target.save()

            with tracer.invoke(corrupted_batch):
                for name, layer in self.model.tracing_layers:
                    target = self.model.get_tracing_target(layer)
                    self.corrupted_out[name] = target.save()
                    self.corrupted_grads[name] = target.grad.save()

            # with tracer.invoke(clean_batch) as invoker_clean:
            #     for layer_name in self.model.tracing_layers:
            #         layer = getattr(self.model, layer_name)
            #         # Get clean attention output for this layer
            #         # across all attention heads
            #         attn_out = layer.attn.c_proj.input
            #         self.clean_out.append(attn_out.save())

            # with tracer.invoke(corrupted_batch) as invoker_corrupted:
            #     # Gather each layer's attention and gradients
            #     for layer_name in self.model.tracing_layers:
            #         layer = getattr(self.model, layer_name)
            #         # Get corrupted attention output for this layer
            #         # across all attention heads
            #         attn_out = layer.attn.c_proj.input
            #         self.corrupted_out.append(attn_out.save())
            #         # save corrupted gradients for attribution patching
            #         self.corrupted_grads.append(attn_out.grad.save())
                output = self.model.get_output_proxy()
                value = self.model.metric(output)
                value.backward()
                # # Let's get the logits for the model's output
                # # for the corrupted run
                # logits_layer = getattr(self.model, self.model.logits_layer)
                # logits = logits_layer.output.save()

                # # Our metric uses tensors saved on cpu, so we
                # # need to move the logits to cpu.
                # value = self.metric(logits.cpu())

                # # We also need to run a backwards pass to
                # # update gradient values
                # value.backward()
        # --- proxies are materialised here ---
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

            self.writer.add_data(
                residual_attr.detach().cpu().numpy()
            )
