from sc2rl.memory.trajectory import Trajectory

class Runner:

    def __init__(self, env, agent, sample_spec, max_n=50000):
        self.env = env
        self.agent = agent
        self.sample_spec = sample_spec
        self.max_n = max_n

    def step(self, env_action):
        return self.env.step(env_action)

    def observe(self):
        return self.env.observe()

    def run_1_episode(self):




