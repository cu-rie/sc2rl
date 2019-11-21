from sc2rl.nn.RelationalGraphNetwork import RelationalGraphNetwork
from sc2rl.config.graph_configs import NODE_ALLY, EDGE_ALLY, EDGE_ENEMY, NODE_ENEMY
from sc2rl.config.ConfigBase import ConfigBase


class RelationalGraphNetworkConfig(ConfigBase):
    def __init__(self, gnn_conf=None):
        super(RelationalGraphNetworkConfig, self).__init__(gnn_conf=gnn_conf)

        self.gnn_conf = {
            'prefix': 'gnn',
            'num_layers': 2,
            'model_dim': 17,
            'num_relations': 3,
            'num_neurons': [64, 64],
            'spectral_norm': False,
            'use_concat': False,
        }


class RelationalGraphNetwork(RelationalGraphNetwork):

    def forward(self, graph, node_feature):
        update_feat = super().forward(graph=graph,
                                      node_feature=node_feature,
                                      update_node_type_indices=[NODE_ALLY],
                                      update_edge_type_indices=[EDGE_ALLY, EDGE_ENEMY])

        return update_feat


if __name__ == "__main__":
    conf = RelationalGraphNetworkConfig()
    gn = RelationalGraphNetwork(**conf.gnn_conf)
    print("done")
