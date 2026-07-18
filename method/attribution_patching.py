import re
import torch
from data.activation_writer import ActivationDataBatch, ActivationWriter, SampleMetadata
from eval.logger import Logger
from tqdm import tqdm
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

class AttributionPatching():
    def __init__(self, config, model, perturbator, dataset, device='cuda'):
        self.config = config
        self.model = model
        assert hasattr(self.model, "tracing_layers"), "Model must have attribute 'tracing_layers'."
        self.tracing_layers = self.model.get_tracing_layers()

        self.max_tracing_batches = config.get("method", {}).get("max_tracing_batches", None)
        self.batch_count = 0

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

    def get_attribution_patching_data(self, batch, unit_test=False):
        clean_batch = batch
        prompts = clean_batch["task"]
        perturbed_prompts = self.perturbator.perturb({"prompt": prompts}).perturbed_prompts

        changed_indices = [i for i, (c, p) in enumerate(zip(prompts, perturbed_prompts)) if c != p]
        if not changed_indices and not unit_test:
            return None, None, None

        perturbed = clean_batch.copy()
        perturbed["task"] = perturbed_prompts
        if unit_test:
            return clean_batch, perturbed, list(range(len(prompts)))  # Use all indices for unit test

        idx = torch.tensor(changed_indices)
        clean_filtered = {k: v[idx] if isinstance(v, torch.Tensor) else [v[i] for i in changed_indices] for k, v in clean_batch.items()}
        perturbed_filtered = {k: v[idx] if isinstance(v, torch.Tensor) else [v[i] for i in changed_indices] for k, v in perturbed.items()}

        return clean_filtered, perturbed_filtered, changed_indices

    def main(self, unit_test=False):
        self.writer.write_config(self.config)
        print("Starting attribution patching. Collecting activations and gradients for each batch in the dataset...")
        for batch in tqdm(self.dataset, desc="Attribution patching", unit="batch"):
            print("Processing next batch...")
            self.activation_tracing(batch, unit_test=unit_test)

            if self._check_tracing_limit(unit_test):
                break
        print("Attribution patching complete. All activations and gradients collected.")


    def _check_tracing_limit(self, unit_test):
        if self.max_tracing_batches is not None and self.batch_count >= self.max_tracing_batches:
            print(f"Reached max tracing batches limit ({self.max_tracing_batches}). Stopping further tracing.")
            return True
        if unit_test and self.batch_count >= 1:
            print("Unit test mode: processed one batch, stopping further tracing.")
            return True
        return False

    def activation_tracing(self, batch, unit_test=False):
        print("Resetting stored activations and gradients...")
        self.reset_collections()
        print("Preparing clean and corrupted batches...")
        clean_batch, corrupted_batch, changed_indices = self.get_attribution_patching_data(batch, unit_test=unit_test)

        if clean_batch is None:
            return
        self.batch_count += 1 # Only increment if we actually processed a batch

        sample_ids = [ f"e{batch['episode_index'][i]}_i{batch['index'][i]}" for i in changed_indices ]

        print("Preprocessing batches...")
        clean_batch_processed = self.model.preprocess_batch(clean_batch)
        corrupted_batch_processed = self.model.preprocess_batch(corrupted_batch)

        print("Tracing clean batch activations...")
        with self.model.trace() as tracer:
            with tracer.invoke(clean_batch_processed):
                for name in self.tracing_layers:
                    target = self.get_tracing_target(name)
                    self.clean_out[name] = target.input.save()

                clean_loss = self.model.output.save()
                self.logger.log_metric("clean_loss", clean_loss, step=self.batch_count)

        print("Tracing corrupted batch activations and gradients...")
        with self.model.trace() as tracer:
            with tracer.invoke(corrupted_batch_processed):
                targets = {}
                for name in self.tracing_layers:
                    target = self.get_tracing_target(name)
                    targ = target.input
                    targets[name] = targ
                    self.corrupted_out[name] = targ.save()
                    targ.retain_grad()

                loss = self.model.output
                loss.backward()

                corrupted_loss = loss.save()
                self.logger.log_metric("corrupted_loss", corrupted_loss, step=self.batch_count)

                for name in self.tracing_layers:
                    self.corrupted_grads[name] = targets[name].grad.save()

        print("Writing sample metadata...")
        for i, sample_id in enumerate(sample_ids):
            try:
                token_key = f"{OBS_LANGUAGE_TOKENS}" if f"{OBS_LANGUAGE_TOKENS}" in clean_batch_processed else "input_ids"
                clean_ids = clean_batch_processed[token_key][i].tolist()
                corrupted_ids = corrupted_batch_processed[token_key][i].tolist()
                if len(clean_ids) == len(corrupted_ids):
                    # fast path: same length, keep the original elementwise check
                    perturbed_tokens = clean_batch_processed[token_key][i] != corrupted_batch_processed[token_key][i]
                    perturbed_token_idxs = torch.where(perturbed_tokens)[0].tolist()
                else:
                    min_len = min(len(clean_ids), len(corrupted_ids))
                    start_idx = next(
                        (i for i in range(min_len) if clean_ids[i] != corrupted_ids[i]),
                        min_len
                    )

                    perturbed_token_idxs = [start_idx]  # single representative index into clean sequence
            except Exception as e:
                print(f"Error while computing perturbed token indices for sample {sample_id}: {e}")
                perturbed_token_idxs = []
            self.writer.add_sample_metadata(SampleMetadata(
                sample_id=sample_id,
                instruction=clean_batch["task"][i],
                corrupt_instruction=corrupted_batch["task"][i],
                perturbed_token_idxs=perturbed_token_idxs,
            ))

        print("Writing traced data to the activation writer...")
        for name, clean in self.clean_out.items():
            if unit_test:
                print(f"Clean shape for {name}: {clean.shape}")
                print(f"Corrupted shape for {name}: {self.corrupted_out[name].shape}")
                print(f"Gradients shape for {name}: {self.corrupted_grads[name].shape}")
            self.writer.add_data(ActivationDataBatch(
                layer=name,
                sample_ids=sample_ids,
                clean=clean,
                corrupt=self.corrupted_out[name],
                gradients=self.corrupted_grads[name]
            ))
        print("Tracing complete.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
