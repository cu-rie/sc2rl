import dgl
import torch

from sc2rl.config.ConfigBase import ConfigBase
from sc2rl.rl.networks.RelationalGraphNetwork import RelationalGraphNetwork
from sc2rl.rl.networks.rnn_encoder import RNNEncoder


class MultiStepInputGraphNetworkConfig(ConfigBase):

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
            'num_relations': 3,
            'num_neurons': [128, 128],
        }
        self.set_configs(self._hist_enc_conf, hist_enc_conf)

        self._curr_enc_conf = {
            'prefix': 'curr_enc_conf',
            'num_layers': 1,
            'model_dim': 17,
            'num_relations': 3,
            'num_neurons': [128, 128],
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


class MultiStepInputGraphNetwork(torch.nn.Module):

    def __init__(self, conf):
        super(MultiStepInputGraphNetwork, self).__init__()
        self.conf = conf  # type: MultiStepInputGraphNetworkConfig

        rnn_conf = conf.hist_rnn_conf
        rnn_type = rnn_conf.pop('rnn_type')
        rnn = getattr(torch.nn, rnn_type)
        self.hist_rnn = rnn(**rnn_conf)

        hist_enc_conf = conf.hist_enc_conf
        self.hist_one_step_enc = RelationalGraphNetwork(**hist_enc_conf)

        self.hist_encoder = RNNEncoder(rnn=self.hist_rnn, one_step_encoder=self.hist_one_step_enc)

        curr_enc_conf = conf.curr_enc_conf
        self.curr_encoder = RelationalGraphNetwork(**curr_enc_conf)

    def forward(self, num_time_steps, hist_graph, hist_feature,
                curr_graph, curr_feature):

        h_enc_out, h_enc_hidden = self.hist_encoder(num_time_steps, hist_graph, hist_feature)
        device = h_enc_out.device

        # recent_hist_enc : slice of the last RNN layer's hidden
        recent_h_enc = h_enc_out[:, -1, :]  # [Batch size x rnn hidden]
        c_enc_out = self.curr_encoder(curr_graph, curr_feature)

        if isinstance(curr_graph, dgl.BatchedDGLGraph):
            c_units = curr_graph.batch_num_nodes
            c_units = torch.tensor(c_units, dtype=torch.long, device=device)
        else:
            c_units = curr_graph.number_of_nodes()
        recent_h_enc = recent_h_enc.repeat_interleave(c_units, dim=0)
        c_encoded_node_feature = torch.cat([recent_h_enc, c_enc_out], dim=1)
        return c_encoded_node_feature
