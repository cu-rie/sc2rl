import torch
from sc2rl.nn.RelationalGraphLayer import RelationalGraphLayer
from sc2rl.config.ConfigBase import ConfigBase


class RelationalGraphNetworkConfig(ConfigBase):
    def __init__(self,
                 gnn_conf=None):
        self._gnn_conf = {
            'prefix': 'gnn',
            'num_layers': 2,
            'model_dim': 17,
            'num_relations': 3,
            'num_neurons': [64, 64]
        }
        self.set_configs(self._gnn_conf, gnn_conf)

    @property
    def gnn_conf(self):
        return self.get_conf(self._gnn_conf)


class RelationalGraphNetwork(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 model_dim,
                 num_relations,
                 num_neurons
                 ):
        super(RelationalGraphNetwork, self).__init__()

        layers = []
        for _ in range(num_layers):
            layer = RelationalGraphLayer(model_dim=model_dim,
                                         num_relations=num_relations,
                                         num_neurons=num_neurons
                                         )
            layers.append(layer)

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, graph, node_feature, update_node_type_indices, update_edge_type_indices):
        for layer in self.layers:
            node_feature = layer(graph, node_feature, update_node_type_indices, update_edge_type_indices)

        return node_feature
