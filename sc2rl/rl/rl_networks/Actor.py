import dgl
import torch

from sc2rl.utils.graph_utils import get_batched_index
from sc2rl.nn.RelationalNetwork import RelationalNetwork
from sc2rl.rl.rl_modules.ActionModules import MoveModule, HoldModule, AttackModule


class Actor(torch.nn.Module):

    def __init__(self,
                 num_layers: int,
                 node_dim: int,
                 global_dim: int,
                 use_hypernet=True,
                 hypernet_input_dim=None,
                 num_relations=None,
                 num_head: int = 1,
                 use_norm=True,
                 neighbor_degree=1,
                 relational_enc_num_neurons=[128, 128],
                 move_dim=4,
                 module_num_neurons=[128],
                 hidden_activation: str = 'tanh',
                 out_activation: str = 'tanh',
                 pooling_op: str = 'relu'):
        super(Actor, self).__init__()
        self.move_dim = move_dim
        self.relational_enc = RelationalNetwork(num_layers=num_layers,
                                                model_dim=node_dim,
                                                use_hypernet=use_hypernet,
                                                hypernet_input_dim=hypernet_input_dim,
                                                num_relations=num_relations,
                                                num_head=num_head,
                                                use_norm=use_norm,
                                                neighbor_degree=neighbor_degree,
                                                num_neurons=relational_enc_num_neurons,
                                                pooling_op=pooling_op)

        module_input_dim = node_dim + global_dim

        self.move_module = MoveModule(node_dim=module_input_dim,
                                      move_dim=move_dim,
                                      num_neurons=module_num_neurons,
                                      hidden_activation=hidden_activation,
                                      out_activation=out_activation)

        self.hold_module = HoldModule(node_dim=module_input_dim,
                                      num_neurons=module_num_neurons,
                                      hidden_activation=hidden_activation,
                                      out_activation=out_activation)

        self.attack_module = AttackModule(node_dim=module_input_dim,
                                          num_neurons=module_num_neurons,
                                          hidden_activation=hidden_activation,
                                          out_activation=out_activation)

    def forward(self, graph, attack_graph, node_feature, global_feature, ally_indices, device):
        """
        :param graph: (dgl.Graph or dgl.BatchedGraph)
        :param attack_graph: (dgl.Graph or dgl.BatchedGraph)
        :param node_feature: (pytorch Tensor) [ (Batched) # Nodes x node_feature dim]
        :param global_feature: (pytorch Tensor) [# Graphs x global_feature dim]
        :param ally_indices: (list of list-like) Each element contains index of allies in the graph
        :param device: (str) where the computation happen.
        """

        node_updated, global_updated = self.relational_enc.forward(graph, node_feature, global_feature, device)

        if type(graph) == dgl.BatchedDGLGraph:
            num_nodes = graph.batch_num_nodes
            num_nodes = torch.tensor(num_nodes)
            global_updated = torch.repeat_interleave(global_updated, num_nodes, dim=0)
            batched_ally_indices = get_batched_index(graph, ally_indices)

        else:
            global_updated = global_updated.repeat((node_feature.shape[0], 1))
            batched_ally_indices = ally_indices

        move_node_feature = torch.cat((node_updated, global_updated), dim=-1)
        hold_node_feature = torch.cat((node_updated, global_updated), dim=-1)
        attack_node_feature = torch.cat((node_updated, global_updated), dim=-1)

        move_argument = self.move_module.forward(graph, move_node_feature, batched_ally_indices)
        hold_argument = self.hold_module.forward(graph, hold_node_feature, batched_ally_indices)
        attack_argument = self.attack_module.forward(attack_graph, attack_node_feature, ally_indices)
        return move_argument, hold_argument, attack_argument

    def compute_probs(self, graph, attack_graph, node_feature, global_feature, ally_indices, device):
        move_arg, hold_arg, attack_arg = self.forward(graph, attack_graph, node_feature, global_feature, ally_indices,
                                                      device)

        # Prepare un-normalized probabilities of attacks
        max_num_enemy = 0
        total_num_units = 0

        num_units = []
        for attack_a in attack_arg[0]:
            num_unit, num_enemy = attack_a.shape[0], attack_a.shape[1]
            total_num_units += num_unit
            num_units.append(num_unit)
            if max_num_enemy <= num_enemy:
                max_num_enemy = num_enemy

        attack_arg_out = torch.zeros(total_num_units, max_num_enemy)
        cum_num_units = 0
        for attack_a, num_unit in zip(attack_arg[0], num_units):
            attack_arg_out[cum_num_units:cum_num_units + num_unit, :attack_a.shape[1]] = attack_a
            attack_arg_out[cum_num_units:cum_num_units + num_unit, attack_a.shape[1]:] = 0
            cum_num_units += num_unit

        unnormed_ps = torch.cat((move_arg, hold_arg, attack_arg_out), dim=-1)
        ps = torch.nn.functional.softmax(unnormed_ps, dim=-1)
        log_ps = torch.log(ps + VERY_SMALL_NUMBER)
        unit_entropy = - torch.sum(log_ps * ps, dim=-1)  # per unit entropy

        log_p_move, log_p_hold, log_p_attack = torch.split(log_ps, [self.move_dim, 1, max_num_enemy], dim=1)
        attack_indicies = attack_arg[1]

        return log_p_move, log_p_hold, log_p_attack, attack_indicies, unit_entropy

    def get_action(self, graph, attack_graph, node_feature, global_feature, ally_indices, device):
        log_p_move, log_p_hold, log_p_attack, attack_indicies, _ = self.compute_probs(graph=graph,
                                                                                      attack_graph=attack_graph,
                                                                                      node_feature=node_feature,
                                                                                      global_feature=global_feature,
                                                                                      ally_indices=ally_indices,
                                                                                      device=device)
        log_probs = torch.cat((log_p_move, log_p_hold, log_p_attack), dim=1)
        probs = torch.exp(log_probs)

        if type(graph) == dgl.BatchedDGLGraph:
            ally_indices, num_targets = get_batched_index(graph, ally_indices, return_num_targets=True)
        else:
            num_targets = len(ally_indices)

        if self.training:  # Sample from categorical dist
            dist = torch.distributions.Categorical(probs=probs[ally_indices, :])
            args = dist.sample()  # tensor

        else:
            ally_probs = probs[ally_indices, :]
            args = ally_probs.argmax(dim=1)  # greedy behavior which does not considers high level probs

        args = torch.split(args, num_targets)  # list of tensors
        return args, attack_indicies
