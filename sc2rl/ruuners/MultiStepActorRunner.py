from sc2rl.ruuners.RunnerBase import RunnerBase


class MultiStepActorRuner(RunnerBase):

    def __init__(self, env, agent, sample_spec, n_steps):
        super(MultiStepActorRuner, self).__init__(env=env, agent=agent, sample_spec=sample_spec)

    def 