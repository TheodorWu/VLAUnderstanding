
from tqdm import tqdm

from method.attribution_patching import AttributionPatching


class AttributionPatchingInference(AttributionPatching):
    def __init__(self, config, model, perturbator, env, device='cuda', **kwargs):
        super().__init__(config, model, perturbator, None, device)
        self.env = env
        self.num_episodes = config.get("method", {}).get("num_episodes", 1)

    def __exit__(self, exc_type, exc_value, traceback):
        self.env.close()
        return False

    def main(self, unit_test=False):
        self.writer.write_config(self.config)
        print("Starting attribution patching. Collecting activations and gradients for each episode...")
        for batch in tqdm(range(self.num_episodes), desc="Attribution patching", unit="episode"):
            print("Processing next episode...")
            self.activation_tracing(batch, unit_test=unit_test)

            if self._check_tracing_limit(unit_test):
                break
        print("Attribution patching complete. All activations and gradients collected.")

    def run_episode(self, unit_test=False):
        obs = self.env.reset()
        done = False
        step_count = 0
        while not done and (self.max_tracing_batches is None or self.batch_count < self.max_tracing_batches):
            action = self.model.act(obs)
            batch = self._transform_obs_to_batch(obs)
            self.activation_tracing(batch, unit_test=unit_test)
            obs, reward, done, info = self.env.step(action)
            step_count += 1
        print(f"Episode finished after {step_count} steps.")

    def _transform_obs_to_batch(self, obs):
        return obs
        # TODO: Check if I need to transform the observation into a batch format expected by the activation tracing. If so, implement the transformation logic here.
        # # Transform the observation into a batch format expected by the activation tracing
        # batch = {
        #     "task": [obs["task"]],
        #     "observation": [obs["observation"]],
        #     "action": [obs["action"]],
        #     "reward": [obs["reward"]],
        #     "done": [obs["done"]],
        # }
        # return batch
