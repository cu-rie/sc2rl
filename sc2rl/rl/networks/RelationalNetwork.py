from sc2rl.nn.RelationalNetwork import RelationalNetwork
from sc2rl.config.graph_configs import (NODE_ALLY, NODE_ENEMY,
                                        EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY)


class RelationalNetwork(RelationalNetwork):

    def forward(self, graph, feature_dict):
        update_feat = super().forward(graph=graph,
                                      feature_dict=feature_dict,
                                      update_node_type_indices=[NODE_ALLY, NODE_ENEMY],
                                      update_edge_type_indices=[EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY])

        return update_feat
