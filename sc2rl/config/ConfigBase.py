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
