
from method.attribution_patching import AttributionPatching


class AttributionPatchingInference(AttributionPatching):
    def __init__(self, config, model, perturbator, env, device='cuda', **kwargs):
        super().__init__(config, model, perturbator, None, device)
        self.env = env

    def __exit__(self, exc_type, exc_value, traceback):
        self.env.close()
        return False

    def main(self, unit_test=False):
        pass

    def run_episode(self, unit_test=False):
        obs = self.env.reset()
        done = False
        step_count = 0
        while not done and (self.max_tracing_batches is None or self.batch_count < self.max_tracing_batches):
            action = self.model.act(obs)
            obs, reward, done, info = self.env.step(action)
            step_count += 1
        print(f"Episode finished after {step_count} steps.")

    def single_step_attribution_patching(self, obs, unit_test=False):
        batch = {"obs": obs}
        clean_batch, perturbed_batch, changed_indices = self.get_attribution_patching_data(batch, unit_test)
        if clean_batch is None:
            return None  # No changes detected, skip this step

        # Here you would add the logic to run the model on both clean and perturbed batches,
        # collect activations and gradients, and perform attribution patching analysis.
        # This is a placeholder for where that logic would go.
        print(f"Changed indices: {changed_indices}")
