
from tqdm import tqdm
import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

from data.activation_writer import ActivationDataBatch, SampleMetadata
from method.attribution_patching import AttributionPatching


class ActivationPatching(AttributionPatching):
    """
    Activation patching method for evaluating the effect of perturbations on model activations.
    """
    def __init__(self, config, model, perturbator, dataset, device='cuda'):
        super().__init__(config, model, perturbator, dataset, device=device)

        config_tracing_layers = config.get("method", {}).get("tracing_layers", None)
        if config_tracing_layers:
            self.tracing_layers = [l for l in self.tracing_layers if l in config_tracing_layers]
        else:
            print("No tracing_layers specified in config, using all available layers.")

    def main(self, unit_test=False):
        self.writer.write_config(self.config)
        print("Starting activation patching. Collecting activations and gradients for each batch in the dataset...")
        for batch in tqdm(self.dataset, desc="Activation patching", unit="batch"):
            print("Processing next batch...")
            self.activation_patching(batch, unit_test=unit_test)

            if self._check_tracing_limit(unit_test):
                break
        print("Activation patching complete. All activations and gradients collected.")

    def activation_patching(self, batch, unit_test=False):
        print("Resetting stored activations...")
        self.reset_collections()
        print("Preparing clean and corrupted batches...")
        clean_batch, corrupted_batch, changed_indices = self.get_attribution_patching_data(batch, unit_test=unit_test)

        if clean_batch is None:
            return
        self.batch_count += 1

        sample_ids = [f"e{batch['episode_index'][i]}_i{batch['index'][i]}" for i in changed_indices]

        print("Preprocessing batches...")
        clean_batch_processed = self.model.preprocess_batch(clean_batch)
        corrupted_batch_processed = self.model.preprocess_batch(corrupted_batch)

        with torch.no_grad():
            # --- Clean pass: cache activations (same as before) ---
            print("Tracing clean batch activations...")
            with self.model.trace() as tracer:
                with tracer.invoke(clean_batch_processed, loss_reduction="sample_mean"):
                    for name in self.tracing_layers:
                        target = self.get_tracing_target(name)
                        self.clean_out[name] = target.input.save()
                    clean_loss = self.model.output.save()
            self.logger.log_metric("clean_loss", clean_loss.mean(), step=self.batch_count)

            # --- Corrupted baseline pass: no patching, no grad ---
            print("Getting corrupted baseline loss...")
            with self.model.trace() as tracer:
                with tracer.invoke(corrupted_batch_processed, loss_reduction="sample_mean"):
                    corrupted_loss = self.model.output.save()
            self.logger.log_metric("corrupted_loss", corrupted_loss.mean(), step=self.batch_count)

            # --- Patched passes: one per layer ---
            print("Running patched forward passes...")
            patched_losses = {}
            for name in self.tracing_layers:
                with self.model.trace() as tracer:
                    with tracer.invoke(corrupted_batch_processed, loss_reduction="sample_mean"):
                        target = self.get_tracing_target(name)
                        # substitute this layer's input with the CLEAN activation
                        target.input[:] = self.clean_out[name]
                        patched_losses[name] = self.model.output.save()

        print("Writing sample metadata...")
        for i, sample_id in enumerate(sample_ids):
            perturbed_tokens = clean_batch_processed[f"{OBS_LANGUAGE_TOKENS}"][i] != corrupted_batch_processed[f"{OBS_LANGUAGE_TOKENS}"][i]
            perturbed_token_idxs = torch.where(perturbed_tokens)[0].tolist()
            self.writer.add_sample_metadata(SampleMetadata(
                sample_id=sample_id,
                instruction=clean_batch["task"][i],
                corrupt_instruction=corrupted_batch["task"][i],
                perturbed_token_idxs=perturbed_token_idxs,
            ))

        print("Writing traced data to the activation writer...")
        for name in self.tracing_layers:
            self.logger.log_metric(f"patched_loss_{name}", patched_losses[name].mean(), step=self.batch_count)
            if unit_test:
                print(f"Clean shape for {name}: {self.clean_out[name].shape}")
                print(f"Corrupted shape for {name}: {corrupted_loss.shape}")
                print(f"Patched shape for {name}: {patched_losses[name].shape}")
                print(f"Clean loss for {name}: {self.clean_out[name].mean()}")
                print(f"Corrupted loss for {name}: {corrupted_loss.mean()}")
                print(f"Patched loss for {name}: {patched_losses[name].mean()}")

            self.writer.add_data(ActivationDataBatch(
                layer=name,
                sample_ids=sample_ids,
                clean=self.clean_out[name],
                patched_loss=patched_losses[name],  # replaces `gradients`
                clean_loss=clean_loss,
                corrupted_loss=corrupted_loss,
            ))
        print("Patching complete.")
