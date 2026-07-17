
from tqdm import tqdm

from method.attribution_patching import AttributionPatching
from eval.logger import Logger


class AttributionPatchingInference(AttributionPatching):
    def __init__(self, config, model, perturbator, env, device='cuda', **kwargs):
        super().__init__(config, model, perturbator, None, device)
        self.max_steps_per_episode = config.get("method", {}).get("max_steps_per_episode", 1000)
        self.env = env
        self.env.env.horizon = self.max_steps_per_episode
        self.num_episodes = config.get("method", {}).get("num_episodes", 1)
        self.logger = Logger()

    def __exit__(self, exc_type, exc_value, traceback):
        self.env.close()
        return False

    def main(self, unit_test=False):
        self.writer.write_config(self.config)
        print("Starting attribution patching. Collecting activations and gradients for each episode...")
        for episode_index in tqdm(range(self.num_episodes), desc="Attribution patching", unit="episode"):
            print("Processing next episode...")
            self.run_episode(unit_test=unit_test, episode_index=episode_index)

            if self._check_tracing_limit(unit_test):
                break
        print("Attribution patching complete. All activations and gradients collected.")

    def run_episode(self, unit_test=False, episode_index=0):
        obs = self.env.reset()
        done = False
        step_count = 0
        for step_count in range(self.max_steps_per_episode):
            if unit_test and episode_index >= 1:
                break

            batch = self.model.transform_obs_to_batch(obs, self.env.language_instruction)
            batch = self.model.preprocess_batch(batch)
            action = self.model.select_action(batch)
            batch['index'] = [step_count]
            batch['episode_index'] = [episode_index]
            self.activation_tracing(batch, unit_test=unit_test)
            print(step_count, self.env.env.done, done)
            obs, reward, done, info = self.env.step(action)

            if done or self.env.env.done:
                self.logger.log_metric("episode_length", step_count + 1, step=episode_index)
                self.logger.log_metric("episode_reward", self.env.env.episode_reward, step=episode_index)
                print(f"Episode {episode_index} finished after {step_count + 1} steps with reward {self.env.env.episode_reward}.")
                break

        else:
            print(f"Warning: Episode {episode_index} reached max steps ({self.max_steps_per_episode}) without termination.")
        print(f"Episode finished after {step_count} steps.")
