==========
Benchmarks
==========

Benchmarks for different models and loss functions on various datasets.

All the experiments are conducted under the settings of:

- 4 GPUs for training, meaning that ``CUDA_VISIBLE_DEVICES=0,1,2,3`` is set for the training scripts
- Total effective batch size of 256. Consider reducing batch size and learning rate if you only have one GPU. See :ref:`gpu-options` for more details.
- Use the default dataset split ``--split 0``, but combine training and validation sets for training models ``--combine-trainval``
- Use the default random seed ``--seed 1``
- Use Euclidean distance directly for evaluation
- Use single-query and single-crop for evaluation
- Full set of evaluation metrics are reported. See :ref:`evaluation-metrics` for more explanations.

.. _cuhk03-benchmark:

^^^^^^
CUHK03
^^^^^^

   ========= ============ ======== ============ ========== ============== ===============
   Net       Loss         Mean AP  CMC allshots CMC cuhk03 CMC market1501 Training Script
   ========= ============ ======== ============ ========== ============== ===============
   Inception Triplet      N/A      N/A          N/A        N/A            N/A
   Inception CrossEntropy 65.8     48.6         73.2       71.0           ``python examples/inception.py -d cuhk03 --combine-trainval --loss xentropy --logs-dir examples/logs/inception-cuhk03-xentropy``
   Inception OIM          71.4     56.0         77.7       76.5           ``python examples/inception.py -d cuhk03 --combine-trainval --loss oim --oim-scalar 20 --logs-dir examples/logs/inception-cuhk03-oim``
   ResNet-50 Triplet      **80.7** **67.9**     **84.3**   **85.0**       ``python examples/resnet.py -d cuhk03 --combine-trainval --num-instances 4 --dropout 0 --loss triplet --optimizer adam --lr 0.0002 --epochs 150 --logs-dir examples/logs/resnet-cuhk03-triplet``
   ResNet-50 CrossEntropy 62.7     44.6         70.8       69.0           ``python examples/resnet.py -d cuhk03 --combine-trainval --loss xentropy --lr 0.1 --logs-dir examples/logs/resnet-cuhk03-xentropy``
   ResNet-50 OIM          72.5     58.2         77.5       79.2           ``python examples/resnet.py -d cuhk03 --combine-trainval --loss oim --oim-scalar 30 --logs-dir examples/logs/resnet-cuhk03-oim``
   ========= ============ ======== ============ ========== ============== ===============

.. _market1501-benchmark:

^^^^^^^^^^
Market1501
^^^^^^^^^^

   ========= ============ ======== ============ ========== ============== ===============
   Net       Loss         Mean AP  CMC allshots CMC cuhk03 CMC market1501 Training Script
   ========= ============ ======== ============ ========== ============== ===============
   Inception Triplet      N/A      N/A          N/A        N/A            N/A
   Inception CrossEntropy 51.8     26.8         57.1       75.8           ``python examples/inception.py -d market1501 --combine-trainval --loss xentropy --logs-dir examples/logs/inception-market1501-xentropy``
   Inception OIM          54.3     30.1         58.3       77.9           ``python examples/inception.py -d market1501 --combine-trainval --loss oim --oim-scalar 20 --logs-dir examples/logs/inception-market1501-oim``
   ResNet-50 Triplet      **67.9** **42.9**     **70.5**   **85.1**       ``python examples/resnet.py -d market1501 --combine-trainval --num-instances 4 --dropout 0 --loss triplet --optimizer adam --lr 0.0002 --epochs 150 --logs-dir examples/logs/resnet-market1501-triplet``
   ResNet-50 CrossEntropy 59.8     35.5         62.8       81.4           ``python examples/resnet.py -d market1501 --combine-trainval --loss xentropy --lr 0.1 --logs-dir examples/logs/resnet-market1501-xentropy``
   ResNet-50 OIM          60.9     37.3         63.6       82.1           ``python examples/resnet.py -d market1501 --combine-trainval --loss oim --oim-scalar 20 --logs-dir examples/logs/resnet-market1501-oim``
   ========= ============ ======== ============ ========== ============== ===============

.. _dukemtmc-benchmark:

^^^^^^^^
DukeMTMC
^^^^^^^^

   ========= ============ ======== ============ ========== ============== ===============
   Net       Loss         Mean AP  CMC allshots CMC cuhk03 CMC market1501 Training Script
   ========= ============ ======== ============ ========== ============== ===============
   Inception Triplet      N/A      N/A          N/A        N/A            N/A
   Inception CrossEntropy 34.0     17.4         39.2       54.4           ``python examples/inception.py -d dukemtmc --combine-trainval --loss xentropy --logs-dir examples/logs/inception-dukemtmc-xentropy``
   Inception OIM          40.6     22.4         45.3       61.7           ``python examples/inception.py -d dukemtmc --combine-trainval --loss oim --oim-scalar 30 --logs-dir examples/logs/inception-dukemtmc-oim``
   ResNet-50 Triplet      **54.6** **34.6**     **57.5**   **73.1**       ``python examples/resnet.py -d dukemtmc --combine-trainval --num-instances 4 --dropout 0 --loss triplet --optimizer adam --lr 0.0002 --epochs 150 --logs-dir examples/logs/resnet-dukemtmc-triplet``
   ResNet-50 CrossEntropy 40.7     23.7         44.3       62.5           ``python examples/resnet.py -d dukemtmc --combine-trainval --loss xentropy --lr 0.1 --logs-dir examples/logs/resnet-dukemtmc-xentropy``
   ResNet-50 OIM          47.4     29.2         50.4       68.1           ``python examples/resnet.py -d dukemtmc --combine-trainval --loss oim --oim-scalar 30 --logs-dir examples/logs/resnet-dukemtmc-oim``
   ========= ============ ======== ============ ========== ============== ===============

.. ATTENTION::
   No test-time augmentation is used. We have fixed a bug in the learning rate
   scheduler for the Triplet loss. Now the result of ResNet-50 with Triplet loss
   is slightly better than the "original" setting in [hermans2017in]_ (Table 4).
