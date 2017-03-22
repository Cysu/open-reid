Training Identification Nets
============================

This example will present how to train nets with identification loss, and
benchmark some models on popular datasets.

.. _training:

Training
--------

After cloning the repository, one may start with training an identification net
on VIPeR, for example

.. code-block:: shell

   python examples/inception.py -d viper -b 64 -j 2 --logs-dir logs/inception-viper-xentropy

This will automatically download the VIPeR dataset and start training, with
batch size of 64 and using two processes for data loading. The training log will
be print to screen as well as saved to ``logs/inception-viper/log.txt``. At the
end of the training, it will evaluate the best model (the one with best
validation performance) on the test set, and report several commonly used
metrics.

For some other datasets, such as CUHK03 or Market1501, one may need to manually
download the datasets, and move them to ``examples/data/cuhk03/raw/`` or
``examples/data/market1501/raw/``. Then run the script with specific dataset

.. code-block:: shell

   # CUHK03
   CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/inception.py -d cuhk03 --loss oim --oim-scalar 1 --logs/inception-cuhk03-oim

   # Or Market1501
   CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/inception.py -d market1501 --loss oim --oim-scalar 15 --logs/inception-market1501-oim

Note that by default we use batch size of 256, which will be distributed into
four GPU cards as specified by ``CUDA_VISIBLE_DEVICES``. If you only have one
GPU card, consider reducing the batch size and the initial learning rate, for
example,

.. code-block:: shell

   # CUHK03
   python examples/inception.py -d cuhk03 -b 64 --lr 0.025 --loss oim --oim-scalar 1 --logs/inception-cuhk03-oim

   # Or Market1501
   python examples/inception.py -d market1501 -b 64 --lr 0.025 --loss oim --oim-scalar 15 --logs/inception-market1501-oim

However, it might lead to slightly worse result.

.. _benchmarks:

Benchmarks
----------

Here we show some benchmark results of different models on various datasets. As
different datasets use different CMC calculations, we include three types of
CMCs, namely allshots, cuhk03, and market1501. See [] for more details.

These benchmarks are trained using four GPU cards. By default, performance are
reported on each dataset with ``--split 0 --seed 1``.

CUHK03
^^^^^^

.. table::
   :column-wrapping: fftttf
   :column-alignment: llrrrl

   ========= ============ ============ ========== ============== ================================
   Net       Loss         CMC allshots CMC cuhk03 CMC market1501 Training Parameters
   ========= ============ ============ ========== ============== ================================
   Inception CrossEntropy 46.7         71.0       68.4           ``python examples/inception.py -d cuhk03 --loss xentropy``
   Inception OIM          55.9         77.0       76.0           ``python examples/inception.py -d cuhk03 --loss oim --oim-scalar 1``
   ResNet-50 CrossEntropy 41.5         68.5       66.0           ``python examples/resnet.py -d cuhk03 --loss xentropy``
   ResNet-50 OIM          53.1         74.7       74.9           ``python examples/resnet.py -d cuhk03 --loss oim --oim-scalar 30``
   ========= ============ ============ ========== ============== ================================

Market1501
^^^^^^^^^^

.. table::
   :column-wrapping: fftttf
   :column-alignment: llrrrl

   ========= ============ ============ ========== ============== ================================
   Net       Loss         CMC allshots CMC cuhk03 CMC market1501 Training Parameters
   ========= ============ ============ ========== ============== ================================
   Inception CrossEntropy 23.0         52.7       71.3           ``python examples/inception.py -d market1501 --loss xentropy``
   Inception OIM          25.8         53.5       73.6           ``python examples/inception.py -d market1501 --loss oim --oim-scalar 10``
   ResNet-50 CrossEntropy 26.0         53.3       73.8           ``python examples/resnet.py -d market1501 --loss xentropy``
   ResNet-50 OIM          28.9         55.8       76.0           ``python examples/resnet.py -d market1501 --loss oim --oim-scalar 30``
   ========= ============ ============ ========== ============== ================================

Duke
^^^^

.. table::
   :column-wrapping: fftttf
   :column-alignment: llrrrl

   ========= ============ ============ ========== ============== ================================
   Net       Loss         CMC allshots CMC cuhk03 CMC market1501 Training Parameters
   ========= ============ ============ ========== ============== ================================
   Inception CrossEntropy 13.8         35.1       50.0           ``python examples/inception.py -d duke --loss xentropy``
   Inception OIM          18.4         40.7       56.1           ``python examples/inception.py -d duke --loss oim --oim-scalar 20``
   ResNet-50 CrossEntropy 18.9         38.5       58.1           ``python examples/resnet.py -d duke --loss xentropy``
   ResNet-50 OIM          23.9         44.7       63.3           ``python examples/resnet.py -d duke --loss oim --oim-scalar 30``
   ========= ============ ============ ========== ============== ================================
