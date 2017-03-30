# Open-ReID

Open-ReID is a lightweight library of person re-identification for research
purpose. It aims to provide a uniform interface for different datasets, a full
set of models and evaluation metrics, as well as examples to reproduce (near)
state-of-the-art results. Open-ReID is mainly based on
[PyTorch](http://pytorch.org/).

## Installation

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

Check about more details at [here](http://open-reid.readthedocs.io/en/latest/examples/training_id.html)
and the [benchmarks](http://open-reid.readthedocs.io/en/latest/examples/benchmarks.html) page.
