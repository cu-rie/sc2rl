import dgl
import torch
from collections import namedtuple, deque

from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.reward_funcs import great_victor_with_kill_bonus
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.memory.n_step_memory import NstepInputMemory
from sc2rl.rl.networks.RelationalNetwork import RelationalNetwork
from sc2rl.rl.networks.rnn_encoder import RNNEncoder
from sc2rl.rl.agents.MultiStepActorCriticAgent import MultiStepActorCriticAgent
from sc2rl.rl.brains.MultiStepActorCriticBrain import MultiStepActorCriticBrain, get_hyper_param_dict
from sc2rl.rl.modules.ActorCritic import ActorCriticModule

from sc2rl.runners.RunnerManager import RunnerConfig, RunnerManager


if __name__ == "__main__":
    map_name = "training_scenario_1"

    node_dim = 20
    rnn_hidden_dim = 32
    rnn_layers = 2

    num_hist_enc_layer = 1
    num_cur_enc_layer = 1

    actor_rnn = torch.nn.GRU(input_size=node_dim,
                             hidden_size=rnn_hidden_dim,
                             num_layers=rnn_layers,
                             batch_first=True)

    actor_one_step_hist_enc = RelationalNetwork(num_layers=num_hist_enc_layer,
                                                model_dim=node_dim,
                                                use_hypernet=False,
                                                hypernet_input_dim=None,
                                                num_relations=3,
                                                num_head=2,
                                                use_norm=True,
                                                neighbor_degree=0,
                                                num_neurons=[128, 128],
                                                pooling_op='relu')

    actor_hist_encoder = RNNEncoder(
        rnn=actor_rnn, one_step_encoder=actor_one_step_hist_enc)

    actor_curr_encoder = RelationalNetwork(num_layers=num_cur_enc_layer,
                                           model_dim=node_dim,
                                           use_hypernet=False,
                                           hypernet_input_dim=None,
                                           num_relations=3,
                                           num_head=2,
                                           use_norm=True,
                                           neighbor_degree=0,
                                           num_neurons=[128, 128],
                                           pooling_op='relu')

    critic_rnn = torch.nn.GRU(input_size=node_dim,
                              hidden_size=rnn_hidden_dim,
                              num_layers=rnn_layers,
                              batch_first=True)

    critic_one_step_hist_enc = RelationalNetwork(num_layers=num_hist_enc_layer,
                                                 model_dim=node_dim,
                                                 use_hypernet=False,
                                                 hypernet_input_dim=None,
                                                 num_relations=3,
                                                 num_head=2,
                                                 use_norm=True,
                                                 neighbor_degree=0,
                                                 num_neurons=[128, 128],
                                                 pooling_op='relu')

    critic_hist_encoder = RNNEncoder(
        rnn=actor_rnn, one_step_encoder=actor_one_step_hist_enc)

    critic_curr_encoder = RelationalNetwork(num_layers=num_cur_enc_layer,
                                            model_dim=node_dim,
                                            use_hypernet=False,
                                            hypernet_input_dim=None,
                                            num_relations=3,
                                            num_head=2,
                                            use_norm=True,
                                            neighbor_degree=0,
                                            num_neurons=[128, 128],
                                            pooling_op='relu')

    actor_critic = ActorCriticModule(node_input_dim=rnn_hidden_dim + node_dim)

    brain_hyper_param = get_hyper_param_dict()
    brain = MultiStepActorCriticBrain(actor_critic=actor_critic,
                                      actor_hist_encoder=actor_hist_encoder,
                                      actor_curr_encoder=actor_curr_encoder,
                                      critic_hist_encoder=critic_hist_encoder,
                                      critic_curr_encoder=critic_curr_encoder,
                                      hyper_params=brain_hyper_param)

    sample_spec = namedtuple('exp_args', ["state", "action", "reward", "next_state", "done"],
                             defaults=tuple([list() for _ in range(4)]))

    num_hist_steps = 5

    buffer = NstepInputMemory(
        N=num_hist_steps, max_n_episodes=100, spec=sample_spec, gamma=1.0, max_traj_len=40)

    agent = MultiStepActorCriticAgent(brain=brain, buffer=buffer)

    config = RunnerConfig(map_name=map_name, reward_func=great_victor_with_kill_bonus,
                          state_proc_func=process_game_state_to_dgl,
                          agent=agent,
                          n_hist_steps=num_hist_steps)

    runner_manager = RunnerManager(config, 1)

    iters = 0
    while iters < 10:
        iters += 1
        runner_manager.sample(10)

        print("fit at {}".format(iters))
        fit_return_dict = agent.fit(
            batch_size=20, hist_num_time_steps=num_hist_steps)

    runner_manager.close()
