from sc2rl.nn.FeedForward import FeedForward
from sc2rl.config.graph_configs import NODE_ALLY

from sc2rl.config.ConfigBase import ConfigBase


class FeedForwardConfig(ConfigBase):
    def __init__(self,
                 mlp_conf=None):
        self._mlp_conf = {
            'prefix': 'mlp_conf',
            'input_dimension': 17,
            'output_dimension': 17,

        }
        self.set_configs(self._mlp_conf, mlp_conf)

    @property
    def mlp_conf(self):
        return self.get_conf(self._mlp_conf)


class FeedForward(FeedForward):
    def __init__(self, conf):
        super(FeedForward, self).__init__(conf=conf, num_node_types=1)

    def forward(self, graph, node_feature):
        graph.ndata['node_feature'] = node_feature
        update_feat = super().forward(graph=graph,
                                      node_feature=node_feature,
                                      update_node_type_indices=[NODE_ALLY])

        return update_feat
