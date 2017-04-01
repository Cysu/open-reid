# Open-ReID

Open-ReID is a lightweight library of person re-identification for research
purpose. It aims to provide a uniform interface for different datasets, a full
set of models and evaluation metrics, as well as examples to reproduce (near)
state-of-the-art results.

## Installation

Install [PyTorch](http://pytorch.org/) (version >= 0.1.11). Although we support
both python2 and python3, we recommend python3 for better performance.

```shell
git clone https://github.com/Cysu/open-reid.git
cd open-reid
python setup.py install
```

## Examples

```shell
python examples/resnet.py -d viper -b 64 -j 2 --loss oim --logs-dir logs/resnet-viper-oim
```

This is just a quick example. VIPeR dataset may not be large enough to train a deep neural network.

Check about more [examples](https://cysu.github.io/open-reid/examples/training_id.html)
and [benchmarks](https://cysu.github.io/open-reid/examples/benchmarks.html).
