from .cuhk01 import CUHK01
from .cuhk03 import CUHK03
from .duke import Duke
from .market1501 import Market1501
from .viper import VIPeR

__factory = {
    'viper': VIPeR,
    'cuhk01': CUHK01,
    'cuhk03': CUHK03,
    'market1501': Market1501,
    'duke': Duke,
}


def get_dataset(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
