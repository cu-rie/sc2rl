from sc2rl.nn.FeedForward import FeedForward
from sc2rl.config.graph_configs import NODE_ALLY

from sc2rl.config.ConfigBase_refac import ConfigBase


class FeedForwardConfig(ConfigBase):
    def __init__(self, mlp_conf=None):
        super(FeedForwardConfig, self).__init__(mlp_conf=mlp_conf)
        self.mlp_conf = {
            'prefix': 'mlp_conf',
            'input_dimension': 17,
            'output_dimension': 1
        }


class FeedForward(FeedForward):
    def __init__(self, conf):
        super(FeedForward, self).__init__(conf=conf, num_node_types=1)

    def forward(self, graph, node_feature):
        update_feat = super().forward(graph=graph,
                                      node_feature=node_feature,
                                      update_node_type_indices=[NODE_ALLY])

        return update_feat


if __name__ == "__main__":
    conf = FeedForwardConfig()
    ff = FeedForward(conf)
    print('done')
