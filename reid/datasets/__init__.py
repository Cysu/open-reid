from .cuhk03 import CUHK03
from .viper import VIPeR

__sets = {
    'viper': VIPeR,
    'cuhk03': CUHK03,
}


def get_dataset(name, root, *args, **kwargs):
    if name not in __sets:
        raise KeyError("Unknown dataset:", name)
    return __sets[name](root, *args, **kwargs)
