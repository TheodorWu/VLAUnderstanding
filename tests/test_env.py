import unittest

class TestLiberoEnv(unittest.TestCase):
    def test_libero_env(self):
        from envs.libero import get_libero
        env, task_description = get_libero(task_id=0)
        self.assertIsNotNone(env)
        self.assertIsNotNone(task_description)
        dummy_action = [0.] * 7
        for step in range(10):
            obs, reward, done, info = env.step(dummy_action)
        env.close()


if __name__ == "__main__":
    # t = TestLiberoEnv()
    # t.setUp()
    # t.test_libero_dataset_loading()
    # t.tearDown()
    unittest.main()
