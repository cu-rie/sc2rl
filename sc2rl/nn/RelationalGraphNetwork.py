import torch
from sc2rl.nn.RelationalGraphLayer import RelationalGraphLayer

from sc2rl.utils.debug_utils import dn


class RelationalGraphNetwork(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 model_dim,
                 num_relations,
                 num_neurons,
                 spectral_norm,
                 use_concat,
                 use_multi_node_types,
                 node_update_types,
                 edge_update_types,
                 use_noisy=False):
        super(RelationalGraphNetwork, self).__init__()

        layers = []
        for _ in range(num_layers):
            layer = RelationalGraphLayer(model_dim=model_dim,
                                         num_relations=num_relations,
                                         num_neurons=num_neurons,
                                         spectral_norm=spectral_norm,
                                         use_concat=use_concat,
                                         use_multi_node_types=use_multi_node_types,
                                         use_noisy=use_noisy)
            layers.append(layer)

        self.layers = torch.nn.ModuleList(layers)
        self.node_update_types = node_update_types
        self.edge_update_types = edge_update_types

    def forward(self, graph, node_feature, update_node_type_indices, update_edge_type_indices):
        for layer in self.layers:
            updated_node_feature = layer(graph, node_feature, update_node_type_indices, update_edge_type_indices)
            node_feature = updated_node_feature

        return node_feature
