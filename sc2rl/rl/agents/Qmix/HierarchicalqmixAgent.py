import dgl
import torch

from sc2rl.rl.modules.HierarchicalMultiStepQnet import HierarchicalMultiStepInputQnet
from sc2rl.rl.brains.QMix.mixer import SubQmixer, SupQmixer
from sc2rl.rl.brains.QMix.HierarchicalqmixBrain import HierarchicalQmixBrain
from sc2rl.memory.n_step_memory import NstepInputMemory

from sc2rl.utils.sc2_utils import nn_action_to_sc2_action
from sc2rl.utils.graph_utils import get_largest_number_of_enemy_nodes

from sc2rl.config.ConfigBase import ConfigBase


class HierarchicalQmixAgentConf(ConfigBase):
    def __init__(self, agent_conf=None, fit_conf=None):
        super(HierarchicalQmixAgentConf, self).__init__(agent_conf=agent_conf,
                                                        fit_conf=fit_conf)
        self.agent_conf = {
            'prefix': 'agent',
            'use_target': True,
            'use_clipped_q': False
        }

        self.fit_conf = {
            'prefix': 'agent_fit',
            'batch_size': 256,
            'hist_num_time_steps': 1
        }


class HierarchicalQmixAgent(torch.nn.Module):

    def __init__(self, conf, qnet_conf, mixer_gnn_conf, mixer_ff_conf, sup_mixer_conf, brain_conf, buffer_conf):
        super(HierarchicalQmixAgent, self).__init__()
        self.conf = conf

        qnet = HierarchicalMultiStepInputQnet(qnet_conf, mixer_gnn_conf, mixer_ff_conf)
        mixer = SupQmixer(input_dim=qnet_conf.qnet_actor_conf['node_input_dim'], conf=sup_mixer_conf)

        if self.conf.agent_conf['use_target']:
            qnet_target = HierarchicalMultiStepInputQnet(qnet_conf, mixer_gnn_conf, mixer_ff_conf)
            mixer_target = SupQmixer(input_dim=qnet_conf.qnet_actor_conf['node_input_dim'], conf=sup_mixer_conf)
        else:
            qnet_target = None
            mixer_target = None

        if self.conf.agent_conf['use_clipped_q']:
            qnet2 = HierarchicalMultiStepInputQnet(qnet_conf, mixer_gnn_conf, mixer_ff_conf)
            mixer2 = SupQmixer(input_dim=qnet_conf.qnet_actor_conf['node_input_dim'], conf=sup_mixer_conf)
        else:
            qnet2 = None
            mixer2 = None

        self.brain = HierarchicalQmixBrain(conf=brain_conf,
                                           qnet=qnet,
                                           mixer=mixer,
                                           qnet_target=qnet_target,
                                           mixer_target=mixer_target,
                                           qnet2=qnet2,
                                           mixer2=mixer2)

        self.buffer = NstepInputMemory(**buffer_conf.memory_conf)

    def get_action(self,
                   hist_graph,
                   curr_graph,
                   tag2unit_dict):

        assert isinstance(curr_graph, dgl.DGLGraph), "get action is designed to work on a single graph!"
        num_time_steps = hist_graph.batch_size
        hist_node_feature = hist_graph.ndata.pop('node_feature')
        curr_node_feature = curr_graph.ndata.pop('node_feature')
        maximum_num_enemy = get_largest_number_of_enemy_nodes([curr_graph])

        nn_actions, info_dict = self.brain.get_action(num_time_steps,
                                                      hist_graph, hist_node_feature,
                                                      curr_graph, curr_node_feature, maximum_num_enemy)

        ally_tags = info_dict['ally_tags']
        enemy_tags = info_dict['enemy_tags']

        sc2_actions = nn_action_to_sc2_action(nn_actions=nn_actions,
                                              ally_tags=ally_tags,
                                              enemy_tags=enemy_tags,
                                              tag2unit_dict=tag2unit_dict)

        hist_graph.ndata['node_feature'] = hist_node_feature
        curr_graph.ndata['node_feature'] = curr_node_feature
        return nn_actions, sc2_actions

    def fit(self, device='cpu'):
        # the prefix 'c' indicates #current# time stamp inputs
        # the prefix 'n' indicates #next# time stamp inputs

        # expected specs:
        # bs = batch_size, nt = hist_num_time_steps
        # 'h_graph' = list of graph lists [[g_(0,0), g_(0,1), ... g_(0,nt)],
        #                                  [g_(1,0), g_(1,1), ..., g_(1,nt)],
        #                                  [g_(2,0), ..., g_(bs, 0), ... g_(bs, nt)]]
        # 'graph' = list of graphs  [g_(0), g_(1), ..., g_(bs)]

        fit_conf = self.conf.fit_conf

        batch_size = fit_conf['batch_size']
        hist_num_time_steps = fit_conf['hist_num_time_steps']

        c_h_graph, c_graph, actions, rewards, n_h_graph, n_graph, dones = self.buffer.sample(batch_size)

        c_maximum_num_enemy = get_largest_number_of_enemy_nodes(c_graph)
        n_maximum_num_enemy = get_largest_number_of_enemy_nodes(n_graph)

        # batching graphs
        list_c_h_graph = [g for L in c_h_graph for g in L]
        list_n_h_graph = [g for L in n_h_graph for g in L]

        c_hist_graph = dgl.batch(list_c_h_graph)
        n_hist_graph = dgl.batch(list_n_h_graph)

        c_curr_graph = dgl.batch(c_graph)
        n_curr_graph = dgl.batch(n_graph)

        # casting actions to one torch tensor
        actions = torch.cat(actions).long()

        # prepare rewards
        rewards = torch.Tensor(rewards)

        # preparing dones
        dones = torch.Tensor(dones)

        if device != 'cpu':
            c_hist_graph.to(torch.device('cuda'))
            n_hist_graph.to(torch.device('cuda'))
            c_curr_graph.to(torch.device('cuda'))
            n_curr_graph.to(torch.device('cuda'))
            actions = actions.to(torch.device('cuda'))
            rewards = rewards.to(torch.device('cuda'))
            dones = dones.to(torch.device('cuda'))

        c_hist_feature = c_hist_graph.ndata.pop('node_feature')
        c_curr_feature = c_curr_graph.ndata.pop('node_feature')

        n_hist_feature = n_hist_graph.ndata.pop('node_feature')
        n_curr_feature = n_curr_graph.ndata.pop('node_feature')

        fit_return_dict = self.brain.fit(num_time_steps=hist_num_time_steps,
                                         c_hist_graph=c_hist_graph,
                                         c_hist_feature=c_hist_feature,
                                         c_curr_graph=c_curr_graph,
                                         c_curr_feature=c_curr_feature,
                                         c_maximum_num_enemy=c_maximum_num_enemy,
                                         n_hist_graph=n_hist_graph,
                                         n_hist_feature=n_hist_feature,
                                         n_curr_graph=n_curr_graph,
                                         n_curr_feature=n_curr_feature,
                                         n_maximum_num_enemy=n_maximum_num_enemy,
                                         actions=actions,
                                         rewards=rewards,
                                         dones=dones)

        return fit_return_dict
