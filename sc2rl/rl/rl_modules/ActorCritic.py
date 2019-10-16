import torch
import dgl
from sc2rl.rl.rl_modules.ActorModule import ActorModule
from sc2rl.config.graph_configs import EDGE_IN_ATTACK_RANGE, NODE_ALLY
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type
from torch.optim.optimizer import


class ActorCriticModule(torch.nn.Module):

    def __init__(self,
                 node_input_dim,
                 actor_out_activation='tanh',
                 critic_out_activation='relu',
                 hidden_activation='relu',
                 move_dim: int = 4,
                 actor_lr=1e-4,
                 critic_lr=1e-3
                 ):
        super(ActorCriticModule, self).__init__()
        self.move_dim = move_dim
        self.actor = ActorModule(node_input_dim, actor_out_activation, hidden_activation)
        self.critic = ActorModule(node_input_dim, critic_out_activation, hidden_activation)
        self.target = ActorModule(node_input_dim, critic_out_activation, hidden_activation)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_optimizer = torch.optim.Adam(self.target.parameters(), lr=critic_lr)

    def forward(self, graph, node_feature, maximum_num_enemy, attack_edge_type_index=EDGE_IN_ATTACK_RANGE):
        pass

    def fit(self, graph, node_feature, maximum_num_enemy):
        exp_q, exp_target_q = self.get_exp_qs(graph, node_feature, maximum_num_enemy)

    def fit_critic(self, graph, node_feature, maximum_num_enemy, next_graph, next_node_feature,
                   next_maximum_num_enemy, rewards):
        cur_q = self.get_q(graph, node_feature, maximum_num_enemy, self.critic)

        with torch.no_grad:
            exp_target_q = self.get_next_exp_qs(next_graph, next_node_feature, next_maximum_num_enemy, self.target)

        target_q = rewards + self.gamma * exp_target_q

        loss = torch.nn.functional.mse_loss(input=cur_q, target=target_q)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return loss.detach().cpu().numpy()

    def fit_actor(self, graph, node_feature, maximum_num_enemy):
        ret_dict = self.actor.compute_probs(graph, node_feature, maximum_num_enemy)
        probs = ret_dict['probs']
        log_p_move = ret_dict['log_p_move']
        log_p_hold = ret_dict['log_p_hold']
        log_p_attack = ret_dict['log_p_attack']


    def get_action(self, *args, **kwargs):
        return self.actor.get_action(*args, **kwargs)

    def get_exp_q(self, graph, node_feature, maximum_num_enemy, net):
        actor_ret_dict = self.actor.compute_probs(graph, node_feature, maximum_num_enemy)
        probs = actor_ret_dict['probs']

        q = self.get_q(graph, node_feature, maximum_num_enemy, net)
        exp_q = q * probs
        return exp_q

    def get_q(self, graph, node_feat, maximum_num_enemy, net):
        ally_node_idx = get_filtered_node_index_by_type(graph, NODE_ALLY)
        q_m, q_h, q_a = net(graph, node_feat, maximum_num_enemy)
        q = torch.cat([q_m, q_h, q_a], dim=-1)
        q_ally = q[ally_node_idx, :]
        return q_ally


if __name__ == "__main__":
    ActorCriticModule(node_input_dim=32)
