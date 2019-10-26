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
                 ):
        super(RelationalGraphLayer, self).__init__()
        self.model_dim = model_dim
        self.num_relations = num_relations

        self.relational_updater = dict()
        for i in range(num_relations):
            relational_updater = MLP(model_dim, model_dim, num_neurons)
            self.relational_updater['updater{}'.format(i)] = relational_updater

        self.node_updater_input_dim = model_dim * (num_relations + 1)

        self.node_updater = MLP(self.node_updater_input_dim, model_dim, num_neurons)

    def forward(self, graph, node_feature, update_node_type_indices, update_edge_type_indices):

        graph.ndata['node_feature'] = node_feature

        message_func = partial(self.message_function, update_edge_type_indices=update_edge_type_indices)
        reduce_func = partial(self.reduce_function, update_edge_type_indices=update_edge_type_indices)
        graph.send_and_recv(graph.edges(), message_func=message_func, reduce_func=reduce_func)
        for ntype_idx in update_node_type_indices:
            node_indices = get_filtered_node_index_by_type(graph, ntype_idx)
            graph.apply_nodes(self.apply_node_function, v=node_indices)

        updated_node_feature = graph.ndata.pop('node_feature')
        graph.ndata.pop('aggregated_node_feature')
        return updated_node_feature

    def message_function(self, edges, update_edge_type_indices):
        src_node_features = edges.src['node_feature']
        edge_types = edges.data['edge_type']

        msg_dict = dict()

        for i in update_edge_type_indices:
            msg = torch.zeros(src_node_features.shape[0], self.model_dim)
            updater = self.relational_updater['updater{}'.format(i)]

            curr_relation_mask = edge_types == i
            curr_relation_pos = torch.arange(src_node_features.shape[0])[curr_relation_mask]
            if curr_relation_mask.sum() == 0:
                pass
            else:
                curr_node_features = src_node_features[curr_relation_mask]
                msg[curr_relation_pos, :] = relu(updater(curr_node_features))
            msg_dict['msg_{}'.format(i)] = msg

        return msg_dict

    def reduce_function(self, nodes, update_edge_type_indices):
        node_feature = nodes.data['node_feature']

        node_enc_input = torch.zeros(node_feature.shape[0], self.node_updater_input_dim)
        node_enc_input[:, :self.model_dim] = relu(node_feature)

        for i in update_edge_type_indices:
            msg = nodes.mailbox['msg_{}'.format(i)]
            reduced_msg = msg.sum(dim=1)
            node_enc_input[:, self.model_dim * (i + 1):self.model_dim * (i + 2)] = reduced_msg

        return {'aggregated_node_feature': node_enc_input}

    def apply_node_function(self, nodes):
        aggregated_node_feature = nodes.data['aggregated_node_feature']
        out = self.node_updater(aggregated_node_feature)
        return {'node_feature': out}
