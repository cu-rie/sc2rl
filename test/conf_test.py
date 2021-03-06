from copy import deepcopy
import warnings


class ConfigBase:

    def __init__(self, **kwargs):
        self._confs = dict()
        for key, arg in kwargs.items():
            if isinstance(arg, dict):
                self._confs[key] = arg

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            if '_confs' in self.__dict__:
                if key in self.__dict__['_confs']:
                    conf = self.__dict__['_confs'][key]
                    value = value
                    value = self.set_configs(value, conf)
        self.__dict__[key] = value

    def __getattribute__(self, item):
        item = super().__getattribute__(item)
        if isinstance(item, dict):
            if 'prefix' in item:
                item = self.get_conf(item)
        return item

    @staticmethod
    def set_configs(target_conf_dict, conf):
        if conf is None:
            return

        for key, val in conf.items():
            if key in target_conf_dict:
                target_conf_dict[key] = val
            else:
                warnings.warn("Unexpected config {} is ignored".format(key))
        return target_conf_dict

    @staticmethod
    def get_conf(conf):
        conf = deepcopy(conf)
        _ = conf.pop('prefix')
        return conf

    def __call__(self, pass_arg=None):
        all_confs = dict()
        confs = deepcopy(self.__dict__)
        _ = confs.pop('_confs')
        for _, conf_vals in confs.items():
            if isinstance(conf_vals, dict):
                conf_vals = deepcopy(conf_vals)
                prefix = conf_vals.pop('prefix')
                for conf_k, conf_v in conf_vals.items():
                    if pass_arg is not None:
                        if conf_k in pass_arg:
                            continue
                    conf_name = '_'.join([prefix, conf_k])
                    all_confs[conf_name] = conf_v
        return all_confs


class Conf(ConfigBase):

    def __init__(self, conf=None, conf2=None):
        super(Conf, self).__init__(conf=conf, conf2=conf2)
        self.conf = {"conf1_key1": "conf1_val1",
                     "conf1_key2": "conf1_val2",
                     'prefix': "prefix_1"}

        self.conf2 = {"conf2_key1": "conf2_val1",
                      "conf2_key2": "conf2_val1",
                      'prefix': "prefix_2"}

        self.not_conf = 2


if __name__ == "__main__":
    b = Conf(conf={"conf1_key1": "conf1_val1_updated"})
    b()
    print(b.conf)
    print(b.conf2)
    print(b.not_conf)
    print(b.__dict__)
