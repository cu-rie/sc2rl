import dgl
import torch

from sc2rl.rl.rl_modules.ActorModule import ActorModule
from sc2rl.config.graph_configs import EDGE_IN_ATTACK_RANGE, NODE_ALLY
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type
from sc2rl.config.nn_configs import VERY_SMALL_NUMBER
from sc2rl.utils.graph_utils import get_index_mapper


class ActorCriticModule(torch.nn.Module):

    def __init__(self,
                 node_input_dim,
                 actor_out_activation='tanh',
                 critic_out_activation='relu',
                 hidden_activation='relu',
                 move_dim: int = 4,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 norm_policy_target: bool = True,
                 entropy_coeff: float = 1e-3
                 ):
        super(ActorCriticModule, self).__init__()
        # sc2 interfacing
        self.move_dim = move_dim

        # RL training
        self.entropy_coeff = entropy_coeff
        self.norm_policy_target = norm_policy_target

        self.actor = ActorModule(node_input_dim, actor_out_activation, hidden_activation)
        self.critic = ActorModule(node_input_dim, critic_out_activation, hidden_activation)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def forward(self, graph, node_feature, maximum_num_enemy, attack_edge_type_index=EDGE_IN_ATTACK_RANGE):
        pass

    def fit(self, graph, node_feature, maximum_num_enemy, next_graph, next_node_feature, next_maximum_num_enemy,
            rewards, target_critic=None):

        actor_loss, entropy_coeff = self.fit_actor(graph, node_feature, maximum_num_enemy)

        critic_loss = self.fit_critic(graph, node_feature, maximum_num_enemy, next_graph, next_node_feature,
                                      next_maximum_num_enemy, rewards, target_critic)
        return actor_loss, critic_loss

    def fit_critic(self, graph, node_feature, maximum_num_enemy, next_graph, next_node_feature,
                   next_maximum_num_enemy, rewards, dones, target_critic=None):
        cur_q = self.get_q(graph, node_feature, maximum_num_enemy, self.critic)

        # The number of allies in the current (batched) graph may differ from the one of the next graph
        target_q = torch.zeros_like(cur_q)

        with torch.no_grad:
            exp_target_q, ally_entropy = self.get_exp_q(next_graph, next_node_feature, next_maximum_num_enemy,
                                                        target_critic)

        next_q = exp_target_q + self.entropy_coeff * ally_entropy  # [# num. next ally_units]
        unsorted_target_q = rewards + self.gamma * next_q * dones

        cur_idx, next_idx = get_index_mapper(graph, next_graph)
        target_q[cur_idx] = unsorted_target_q[next_idx]

        loss = torch.nn.functional.mse_loss(input=cur_q, target=target_q)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return loss.detach().cpu().numpy()

    #
    # def fit_critic_zero_ally(self):
    #     # TODO
    #     pass

    def fit_actor(self, graph, node_feature, maximum_num_enemy):
        ret_dict = self.actor.compute_probs(graph, node_feature, maximum_num_enemy)
        ally_ps = ret_dict['probs']

        log_p_move = ret_dict['log_p_move']
        log_p_hold = ret_dict['log_p_hold']
        log_p_attack = ret_dict['log_p_attack']

        ally_log_ps = torch.cat([log_p_move, log_p_hold, log_p_attack], dim=-1)

        ally_qs = self.get_q(graph, node_feature, maximum_num_enemy)

        ally_vs = (ally_qs * ally_ps).sum(1).detach()

        policy_targets = ally_qs - ally_vs

        if self.norm_policy_target:
            policy_targets = (policy_targets - policy_targets.mean()) / policy_targets.std()

        unmasked_loss = ally_log_ps * (self.entropy_coeff * ally_log_ps - policy_targets)
        loss_mask = (ally_log_ps > torch.log(torch.tensor(VERY_SMALL_NUMBER))).float()
        loss = (unmasked_loss * loss_mask).sum() / loss_mask.sum()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.detach().cpu().numpy(), self.entropy_coeff

    def get_action(self, *args, **kwargs):
        return self.actor.get_action(*args, **kwargs)

    def get_exp_q(self, graph, node_feature, maximum_num_enemy, critic=None):
        actor_ret_dict = self.actor.compute_probs(graph, node_feature, maximum_num_enemy)
        probs = actor_ret_dict['probs']
        ally_entropy = actor_ret_dict['ally_entropy']

        q = self.get_q(graph, node_feature, maximum_num_enemy, critic)
        exp_q = q * probs
        return exp_q, ally_entropy

    def get_q(self, graph, node_feature, maximum_num_enemy, critic=None):
        if critic is None:
            critic = self.critic
        ally_node_idx = get_filtered_node_index_by_type(graph, NODE_ALLY)
        q_m, q_h, q_a = critic(graph, node_feature, maximum_num_enemy)
        q = torch.cat([q_m, q_h, q_a], dim=-1)
        q_ally = q[ally_node_idx, :]
        return q_ally


if __name__ == "__main__":
    ActorCriticModule(node_input_dim=32)
