from .cuhk03 import CUHK03
from .market1501 import Market1501
from .viper import VIPeR

__factory = {
    'viper': VIPeR,
    'cuhk03': CUHK03,
    'market1501': Market1501,
}


def get_dataset(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
