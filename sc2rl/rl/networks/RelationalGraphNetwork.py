from sc2rl.nn.RelationalGraphNetwork import RelationalGraphNetwork
from sc2rl.config.graph_configs import (NODE_ALLY, NODE_ENEMY,
                                        EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY)
from sc2rl.config.ConfigBase import ConfigBase


class RelationalGraphNetworkConfig(ConfigBase):
    def __init__(self, gnn_conf=None):
        super(RelationalGraphNetworkConfig, self).__init__(gnn_conf=gnn_conf)

        self.gnn_conf = {
            'prefix': 'gnn',
            'num_layers': 2,
            'model_dim': 17,
            'num_relations': 4,
            'num_neurons': [64, 64],
            'spectral_norm': False,
            'use_concat': False,
            'use_multi_node_types': False,
            'node_update_types': [NODE_ALLY, NODE_ENEMY],
            'edge_update_types': [EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY],
            'use_noisy': False
        }


class RelationalGraphNetwork(RelationalGraphNetwork):

    def forward(self, graph, node_feature):
        update_feat = super().forward(graph=graph,
                                      node_feature=node_feature,
                                      update_node_type_indices=self.node_update_types,
                                      update_edge_type_indices=self.edge_update_types)

        return update_feat


if __name__ == "__main__":
    conf = RelationalGraphNetworkConfig()
    gn = RelationalGraphNetwork(**conf.gnn_conf)
    print("done")
