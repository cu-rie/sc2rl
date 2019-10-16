import torch
import dgl
from sc2rl.rl.rl_modules.ActorModule import ActorModule
from sc2rl.config.graph_configs import EDGE_IN_ATTACK_RANGE, NODE_ALLY
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type


class ActorCriticModule(torch.nn.Module):

    def __init__(self,
                 node_input_dim,
                 actor_out_activation='tanh',
                 critic_out_activation='relu',
                 hidden_activation='relu',
                 move_dim: int = 4,
                 ):
        super(ActorCriticModule, self).__init__()
        self.move_dim = move_dim
        self.actor = ActorModule(node_input_dim, actor_out_activation, hidden_activation)
        self.critic = ActorModule(node_input_dim, critic_out_activation, hidden_activation)

    def forward(self, graph, node_feature, maximum_num_enemy, attack_edge_type_index=EDGE_IN_ATTACK_RANGE):
        exp_Q = self.get_exp_Qs(graph, node_feature, maximum_num_enemy)


    def get_action(self, *args, **kwargs):
        return self.actor.get_action(*args, **kwargs)

    def get_exp_Qs(self, graph, node_feature, maximum_num_enemy):
        actor_ret_dict = self.actor.compute_probs(graph, node_feature, maximum_num_enemy)
        move_Q, hold_Q, attack_Q = self.critic(graph, node_feature, maximum_num_enemy)
        Q = torch.cat([move_Q, hold_Q, attack_Q], dim=-1)

        ally_node_idx = get_filtered_node_index_by_type(graph, NODE_ALLY)
        ally_Qs = Q[ally_node_idx, :]
        probs = actor_ret_dict['probs']
        return ally_Qs * probs


if __name__ == "__main__":
    ActorCriticModule(node_input_dim=32)
