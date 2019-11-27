from copy import deepcopy

import torch
import numpy as np

from sc2rl.rl.modules.MultiStepInputQnet import MultiStepInputQnet
from sc2rl.rl.modules.HierarchicalQnetActor import HierarchicalQnetActor
from sc2rl.rl.brains.QMix.mixer import SubQmixer, Soft_SubQmixer

from sc2rl.config.ConfigBase import ConfigBase
from sc2rl.config.graph_configs import EDGE_IN_ATTACK_RANGE, EDGE_ENEMY


class HierarchicalMultiStepInputQnetConfig(ConfigBase):

    def __init__(self,
                 multi_step_input_qnet_conf=None,
                 qnet_actor_conf=None,
                 mixer_conf=None):
        super(HierarchicalMultiStepInputQnetConfig, self).__init__(
            multi_step_input_qnet_conf=multi_step_input_qnet_conf,
            qnet_actor_conf=qnet_actor_conf,
            mixer_conf=mixer_conf)

        self.multi_step_input_qnet_conf = {
            'prefix': 'multi_step_input_qnet_conf',
            'use_attention': False,
            'eps': 1.0,
            'exploration_method': 'eps_greedy',
        }

        self.qnet_actor_conf = {
            'prefix': 'qnet_actor_conf',
            'node_input_dim': 17,
            'out_activation': None,
            'hidden_activation': 'mish',
            'num_neurons': [64, 64],
            'spectral_norm': False,
            'num_groups': 3,
            'pooling_op': 'softmax',
            'use_concat_input': False,
            'init_node_dim': 17,
            'pooling_init': None,
            'attack_edge_type_index': EDGE_ENEMY,
            'use_hold': True,
            'use_tanh': False,
            'use_noisy': False
        }

        self.mixer_conf = {
            'prefix': 'mixer_conf',
            'rectifier': 'abs',
            'use_attention': False,
        }


class HierarchicalMultiStepInputQnet(MultiStepInputQnet):

    def __init__(self, conf, mixer_gnn_conf, mixer_ff_conf, soft_assignment=False):
        super(HierarchicalMultiStepInputQnet, self).__init__(conf=conf)
        qnet_actor_conf = conf.qnet_actor_conf
        qnet_actor_conf['node_input_dim'] = self.multi_step_input_net.out_dim
        self.qnet = HierarchicalQnetActor(qnet_actor_conf)

        mixer_rectifier = conf.mixer_conf['rectifier']
        mixer_attn = conf.mixer_conf['use_attention']

        self.mixers = torch.nn.ModuleDict()
        if soft_assignment:
            # base_mixer = Soft_SubQmixer(mixer_gnn_conf, mixer_ff_conf, 0, mixer_rectifier)
            for i in range(conf.qnet_actor_conf['num_groups']):
                mixer = Soft_SubQmixer(mixer_gnn_conf, mixer_ff_conf, i, mixer_rectifier, mixer_attn)
                # mixer = deepcopy(base_mixer)
                # mixer.target_assignment = i
                self.mixers['mixer_{}'.format(i)] = mixer
        else:
            for i in range(conf.qnet_actor_conf['num_groups']):
                mixer = SubQmixer(mixer_gnn_conf, mixer_ff_conf, i, mixer_rectifier)
                self.mixers['mixer_{}'.format(i)] = mixer
