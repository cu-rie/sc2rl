from sc2rl.nn.RelationalNetwork import RelationalNetwork
from sc2rl.config.graph_configs import (NODE_ALLY, NODE_ENEMY,
                                        EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY)
from sc2rl.config.ConfigBase import ConfigBase


class RelationalNetworkConfig(ConfigBase):

    def __init__(self, gnn_conf=None):
        super(RelationalNetworkConfig, self).__init__(gnn_conf=gnn_conf)

        self.gnn_conf = {
            'prefix': 'attn_gnn',
            'num_layers': 1,
            'model_dim': 17,
            'use_hypernet': False,
            'hypernet_input_dim': None,
            'num_relations': 3,
            'num_head': 2,
            'use_norm': True,
            'neighbor_degree': 0,
            'num_neurons': [64, 64],
            'pooling_op': 'relu'
        }


class RelationalNetwork(RelationalNetwork):

    def forward(self, graph, feature_dict):
        update_feat = super().forward(graph=graph,
                                      feature_dict=feature_dict,
                                      update_node_type_indices=[NODE_ALLY, NODE_ENEMY],
                                      update_edge_type_indices=[EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY])

        return update_feat
