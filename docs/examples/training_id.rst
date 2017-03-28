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
