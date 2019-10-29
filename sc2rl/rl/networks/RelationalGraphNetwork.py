from sc2rl.nn.RelationalGraphNetwork import RelationalGraphNetwork
from sc2rl.config.graph_configs import NODE_ALLY, EDGE_ALLY, EDGE_ENEMY



class RelationalGraphNetwork(RelationalGraphNetwork):

    def forward(self, graph, node_feature):
        update_feat = super().forward(graph=graph,
                                      node_feature=node_feature,
                                      update_node_type_indices=[NODE_ALLY],
                                      update_edge_type_indices=[EDGE_ALLY, EDGE_ENEMY])

        return update_feat
