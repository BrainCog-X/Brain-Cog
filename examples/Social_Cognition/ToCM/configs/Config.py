from collections.abc import Iterable


# train->agent_configs = [ToCMControllerConfig(ToCMConfig),] -> class ToCMConfig(Config) -> Config
class Config:
    def __init__(self):
        pass

    def to_dict(self, prefix=""):
        res_dict = dict()
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                res_dict.update(value.to_dict(prefix + str(key) + "_"))
            elif isinstance(value, Iterable):
                if value and isinstance(value[0], Config):
                    for i, v in enumerate(value):
                        res_dict.update(v.to_dict(prefix + str(key) + str(i) + "_"))
                else:
                    res_dict[prefix + str(key)] = value
            else:
                res_dict[prefix + str(key)] = value
        return res_dict
