import warnings
from copy import deepcopy


class ConfigBase:

    @staticmethod
    def set_configs(target_conf_dict, conf):
        if conf is None:
            return

        for key, val in conf:
            if key in target_conf_dict:
                target_conf_dict[key] = val
            else:
                warnings.warn("Unexpected config {} is ignored".format(key))

    @staticmethod
    def get_conf(conf):
        conf = deepcopy(conf)
        _ = conf.pop('prefix')
        return conf

    def __call__(self, pass_arg=None):
        all_confs = dict()
        confs = self.__dict__
        for _, conf_vals in confs.items():
            _conf_vals = deepcopy(conf_vals)
            prefix = _conf_vals.pop('prefix')
            for conf_k, conf_v in _conf_vals.items():
                if pass_arg is not None:
                    if conf_k in pass_arg:
                        continue
                if conf_k == 'prefix':
                    continue
                conf_name = '_'.join([prefix, conf_k])
                all_confs[conf_name] = conf_v
        return all_confs
