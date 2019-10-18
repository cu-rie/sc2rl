from sc2rl.ruuners.RunnerBase import RunnerBase
from sc2rl.utils.HistoryManagers import HistoryManager
from

class MultiStepActorRunner(RunnerBase):

    def __init__(self, env, agent, sample_spec, n_steps):
        super(MultiStepActorRunner, self).__init__(env=env, agent=agent, sample_spec=sample_spec)
        self.history_manager = HistoryManager(n_hist_steps=n_steps, init_graph=None)

    def run_1_episode(self):

        self.history_manager.reset()
