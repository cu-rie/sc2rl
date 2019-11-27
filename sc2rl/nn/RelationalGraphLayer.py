import torch
from torch.nn.functional import relu
from sc2rl.nn.MultiLayerPerceptron import MultiLayerPerceptron as MLP
from functools import partial
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type


class RelationalGraphLayer(torch.nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_relations: int,
                 num_neurons: list = [64, 64],
                 spectral_norm: bool = False,
                 use_concat: bool = False,
                 use_multi_node_types: bool = False,
                 num_node_types: int = 2,
                 use_noisy=False):
        super(RelationalGraphLayer, self).__init__()
        self.model_dim = model_dim
        self.num_relations = num_relations

        self.relational_updater = dict()

        self.use_concat = use_concat
        if self.use_concat:
            relational_encoder_input_dim = 2 * model_dim
            node_updater_input_dim = model_dim * (num_relations + 2)
        else:
            relational_encoder_input_dim = model_dim
            node_updater_input_dim = model_dim * (num_relations + 1)

        for i in range(num_relations):
            relational_updater = MLP(relational_encoder_input_dim, model_dim, num_neurons, spectral_norm, use_noisy=use_noisy)
            self.relational_updater['updater{}'.format(i)] = relational_updater
        self.relational_updater = torch.nn.ModuleDict(self.relational_updater)

        self.node_updater_input_dim = node_updater_input_dim
        self.use_multi_node_types = use_multi_node_types
        if not use_multi_node_types:
            self.node_updater = MLP(self.node_updater_input_dim, model_dim, num_neurons, spectral_norm, use_noisy=use_noisy)
        else:
            self.node_updater = torch.nn.ModuleDict()
            self.node_updater_input_dim = node_updater_input_dim
            for i in range(num_node_types):
                node_updater = MLP(self.node_updater_input_dim, model_dim, num_neurons, spectral_norm, use_noisy=use_noisy)
                self.node_updater['updater{}'.format(i)] = node_updater

    def forward(self, graph, node_feature, update_node_type_indices, update_edge_type_indices):
        if self.use_concat:
            graph.ndata['node_feature'] = torch.cat([node_feature, graph.ndata['init_node_feature']], dim=1)
        else:
            graph.ndata['node_feature'] = node_feature

        message_func = partial(self.message_function, update_edge_type_indices=update_edge_type_indices)
        reduce_func = partial(self.reduce_function, update_edge_type_indices=update_edge_type_indices)
        graph.send_and_recv(graph.edges(), message_func=message_func, reduce_func=reduce_func)

        if not self.use_multi_node_types:  # default behavior
            for ntype_idx in update_node_type_indices:
                node_indices = get_filtered_node_index_by_type(graph, ntype_idx)
                graph.apply_nodes(self.apply_node_function, v=node_indices)

        else:  # testing
            for ntype_idx in update_node_type_indices:
                node_indices = get_filtered_node_index_by_type(graph, ntype_idx)
                node_updater = self.node_updater['updater{}'.format(ntype_idx)]
                apply_node_func = partial(self.apply_node_function_multi_type, updater=node_updater)
                graph.apply_nodes(apply_node_func, v=node_indices)

        updated_node_feature = graph.ndata.pop('updated_node_feature')
        _ = graph.ndata.pop('aggregated_node_feature')
        _ = graph.ndata.pop('node_feature')
        return updated_node_feature

    def message_function(self, edges, update_edge_type_indices):
        src_node_features = edges.src['node_feature']
        edge_types = edges.data['edge_type']

        device = src_node_features.device

        msg_dict = dict()
        for i in update_edge_type_indices:
            msg = torch.zeros(src_node_features.shape[0], self.model_dim, device=device)
            updater = self.relational_updater['updater{}'.format(i)]

            curr_relation_mask = edge_types == i
            curr_relation_pos = torch.arange(src_node_features.shape[0])[curr_relation_mask]
            if curr_relation_mask.sum() == 0:
                msg_dict['msg_{}'.format(i)] = msg
            else:
                curr_node_features = src_node_features[curr_relation_mask]
                msg[curr_relation_pos, :] = relu(updater(curr_node_features))
                msg_dict['msg_{}'.format(i)] = msg
        return msg_dict

    def reduce_function(self, nodes, update_edge_type_indices):
        node_feature = nodes.data['node_feature']
        device = node_feature.device

        node_enc_input = torch.zeros(node_feature.shape[0], self.node_updater_input_dim, device=device)
        if self.use_concat:
            node_enc_input[:, :self.model_dim * 2] = relu(node_feature)
            start_index = 2
        else:
            node_enc_input[:, :self.model_dim] = relu(node_feature)
            start_index = 1

        for i in update_edge_type_indices:
            msg = nodes.mailbox['msg_{}'.format(i)]
            reduced_msg = msg.sum(dim=1)
            node_enc_input[:, self.model_dim * (i + start_index):self.model_dim * (i + start_index + 1)] = reduced_msg

        return {'aggregated_node_feature': node_enc_input}

    def apply_node_function(self, nodes):
        aggregated_node_feature = nodes.data['aggregated_node_feature']
        out = self.node_updater(aggregated_node_feature)
        return {'updated_node_feature': out}

    def apply_node_function_multi_type(self, nodes, updater):
        aggregated_node_feature = nodes.data['aggregated_node_feature']
        out = updater(aggregated_node_feature)
        return {'updated_node_feature': out}
