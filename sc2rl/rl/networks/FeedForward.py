from sc2rl.nn.FeedForward import FeedForward
from sc2rl.config.graph_configs import NODE_ALLY


class FeedForward(FeedForward):

    def forward(self, graph, node_feature):
        graph.ndata['node_feature'] = node_feature
        update_feat = super().forward(graph=graph,
                                      node_feature=node_feature,
                                      update_node_type_indices=[NODE_ALLY])

        return update_feat
