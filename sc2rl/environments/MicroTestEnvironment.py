from sc2 import Race
from sc2.player import Bot
from sc2rl.environments.EnvironmentBase import SC2EnvironmentBase
from sc2rl.environments.SC2BotAI import SimpleSC2BotAI

from sc2rl.config.nn_configs import VERY_LARGE_NUMBER


class MicroTestEnvironment(SC2EnvironmentBase):

    def __init__(self, map_name, reward_func, state_proc_func, realtime=False, max_steps=10000):
        allies = Bot(Race.Terran, SimpleSC2BotAI())
        super(MicroTestEnvironment, self).__init__(map_name=map_name,
                                                   allies=allies,
                                                   realtime=realtime)
        self.max_steps = max_steps
        self.step_counter = 0

        self.reward_func = reward_func
        self.state_proc_func = state_proc_func
        self.prev_health = VERY_LARGE_NUMBER
        self.curr_health = VERY_LARGE_NUMBER

    def reset(self):
        sc2_game_state = self._reset()
        return self.state_proc_func(sc2_game_state)

    def observe(self):
        sc2_game_state = self._observe()
        return self.state_proc_func(sc2_game_state)

    def _check_done(self, sc2_game_state):
        num_allies = len(sc2_game_state.units.owned)
        num_enemies = len(sc2_game_state.units.enemy)
        cur_health = 0
        for u in sc2_game_state.units:
            cur_health += u.health
        self.curr_health = cur_health

        done_increase = num_allies == 0 or num_enemies == 0

        if self.prev_health < self.curr_health:
            done_zero_units = True
        else:
            done_zero_units = False
        self.prev_health = self.curr_health

        return done_increase or done_zero_units

    def step(self, action):
        self.step_counter += 1
        sc2_cur_state = self._observe()
        sc2_next_state, _ = self._step(action_args=action)

        # additional routine for checking done!
        # Done checking behaviour of the variants of 'MicroTest' are different from the standard checking done routine.
        done = self._check_done(sc2_next_state)

        cur_state = self.state_proc_func(sc2_cur_state)
        next_state = self.state_proc_func(sc2_next_state)
        reward = self.reward_func(cur_state, next_state, done)

        if done:  # Burn few remaining frames
            self.burn_last_frames()
            if self.step_counter >= self.max_steps:
                _ = self.reset()

        return next_state, reward, done

    def burn_last_frames(self):
        while True:
            self.step_counter += 1
            sc2_cur_state = self._observe()
            done = self._check_done(sc2_cur_state)
            if not done:
                _, _ = self._step(action_args=None)
                break
            else:
                _, _ = self._step(action_args=None)


if __name__ == "__main__":

    def reward_func(s, ns):
        return 1


    def _convert_nn_action_to_sc2_action(self, nn_action, graph):
        pass


    map_name = "training_scenario_1"
    test_reward_func = reward_func
    test_sate_proc_func = state_proc_func
    done_cnt = 0

    env = MicroTestEnvironment(map_name, test_reward_func, test_sate_proc_func)
    while True:
        cur_state = env.observe()
        next_state, reward, done = env.step(action=None)
        if done:
            done_cnt += 1
            if done_cnt >= 10:
                break

    env.close()

    print("We are in the end game.")
