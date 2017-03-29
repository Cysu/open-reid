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
python examples/inception.py -d viper -b 64 -j 2 --loss xentropy --logs-dir logs/inception-viper-xentropy
```

This is just a quick demo, which may not have very good performance.
