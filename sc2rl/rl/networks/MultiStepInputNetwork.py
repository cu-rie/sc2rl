import dgl
import torch

from sc2rl.config.ConfigBase import ConfigBase
from sc2rl.rl.networks.RelationalNetwork import RelationalNetwork
from sc2rl.rl.networks.rnn_encoder import RNNEncoder


class MultiStepInputNetworkConfig(ConfigBase):

    def __init__(self,
                 hist_rnn_conf=None,
                 hist_enc_conf=None,
                 curr_enc_conf=None):
        self._hist_rnn_conf = {
            'prefix': 'hist_rnn',
            'rnn_type': 'GRU',
            'input_size': 17,
            'hidden_size': 32,
            'num_layers': 2,
            'batch_first': True
        }
        self.set_configs(self._hist_rnn_conf, hist_rnn_conf)

        self._hist_enc_conf = {
            'prefix': 'hist_enc_conf',
            'num_layers': 1,
            'model_dim': 17,
            'use_hypernet': False,
            'hypernet_input_dim': None,
            'num_relations': 3,
            'num_head': 2,
            'use_norm': True,
            'neighbor_degree': 0,
            'num_neurons': [128, 128],
            'pooling_op': 'relu'
        }
        self.set_configs(self._hist_enc_conf, hist_enc_conf)

        self._curr_enc_conf = {
            'prefix': 'curr_enc_conf',
            'num_layers': 1,
            'model_dim': 17,
            'use_hypernet': False,
            'hypernet_input_dim': None,
            'num_relations': 3,
            'num_head': 2,
            'use_norm': True,
            'neighbor_degree': 0,
            'num_neurons': [128, 128],
            'pooling_op': 'relu'
        }
        self.set_configs(self._curr_enc_conf, curr_enc_conf)

    @property
    def hist_rnn_conf(self):
        return self.get_conf(self._hist_rnn_conf)

    @property
    def hist_enc_conf(self):
        return self.get_conf(self._hist_enc_conf)

    @property
    def curr_enc_conf(self):
        return self.get_conf(self._curr_enc_conf)


class MultiStepInputNetwork(torch.nn.Module):

    def __init__(self, conf):
        super(MultiStepInputNetwork, self).__init__()
        self.conf = conf  # type: MultiStepInputNetworkConfig

        rnn_conf = conf.hist_rnn_conf
        rnn_type = rnn_conf.pop('rnn_type')
        rnn = getattr(torch.nn, rnn_type)
        self.hist_rnn = rnn(**rnn_conf)

        hist_enc_conf = conf.hist_enc_conf
        self.hist_one_step_enc = RelationalNetwork(**hist_enc_conf)

        self.hist_encoder = RNNEncoder(rnn=self.hist_rnn, one_step_encoder=self.hist_one_step_enc)

        curr_enc_conf = conf.curr_enc_conf
        self.curr_encoder = RelationalNetwork(**curr_enc_conf)
        self.out_dim = curr_enc_conf['model_dim'] + rnn_conf['hidden_size']

    def forward(self, num_time_steps, hist_graph, hist_feature,
                curr_graph, curr_feature):

        h_enc_out, h_enc_hidden = self.hist_encoder(num_time_steps, hist_graph, hist_feature)

        # recent_hist_enc : slice of the last RNN layer's hidden
        recent_h_enc = h_enc_out[:, -1, :]  # [Batch size x rnn hidden]
        c_enc_out = self.curr_encoder(curr_graph, curr_feature)

        if isinstance(curr_graph, dgl.BatchedDGLGraph):
            c_units = curr_graph.batch_num_nodes
            c_units = torch.Tensor(c_units).long()
            recent_h_enc = recent_h_enc.repeat_interleave(c_units, dim=0)
        else:
            c_unit = curr_graph.number_of_nodes()
            recent_h_enc = recent_h_enc.repeat_interleave(c_unit, dim=0)
        c_encoded_node_feature = torch.cat([recent_h_enc, c_enc_out], dim=1)
        return c_encoded_node_feature
