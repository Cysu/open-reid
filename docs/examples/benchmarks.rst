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
   Inception Softmax      65.8     48.6         73.2       71.0           ``python examples/softmax_loss.py -d cuhk03 -a inception --combine-trainval --epochs 70 --logs-dir examples/logs/softmax-loss/cuhk03-inception``
   Inception OIM          71.4     56.0         77.7       76.5           ``python examples/oim_loss.py -d cuhk03 -a inception --combine-trainval --oim-scalar 20 --epochs 70 --logs-dir examples/logs/oim-loss/cuhk03-inception``
   ResNet-50 Triplet      **80.7** **67.9**     **84.3**   **85.0**       ``python examples/triplet_loss.py -d cuhk03 -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/cuhk03-resnet50``
   ResNet-50 Softmax      62.7     44.6         70.8       69.0           ``python examples/softmax_loss.py -d cuhk03 -a resnet50 --combine-trainval --logs-dir examples/logs/softmax-loss/cuhk03-resnet50``
   ResNet-50 OIM          72.5     58.2         77.5       79.2           ``python examples/oim_loss.py -d cuhk03 -a resnet50 --combine-trainval --oim-scalar 30 --logs-dir examples/logs/oim-loss/cuhk03-resnet50``
   ========= ============ ======== ============ ========== ============== ===============

.. _market1501-benchmark:

^^^^^^^^^^
Market1501
^^^^^^^^^^

   ========= ============ ======== ============ ========== ============== ===============
   Net       Loss         Mean AP  CMC allshots CMC cuhk03 CMC market1501 Training Script
   ========= ============ ======== ============ ========== ============== ===============
   Inception Triplet      N/A      N/A          N/A        N/A            N/A
   Inception Softmax      51.8     26.8         57.1       75.8           ``python examples/softmax_loss.py -d market1501 -a inception --combine-trainval --epochs 70 --logs-dir examples/logs/softmax-loss/market1501-inception``
   Inception OIM          54.3     30.1         58.3       77.9           ``python examples/oim_loss.py -d market1501 -a inception --combine-trainval --oim-scalar 20 --epochs 70 --logs-dir examples/logs/oim-loss/market1501-inception``
   ResNet-50 Triplet      **67.9** **42.9**     **70.5**   **85.1**       ``python examples/triplet_loss.py -d market1501 -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/market1501-resnet50``
   ResNet-50 Softmax      59.8     35.5         62.8       81.4           ``python examples/softmax_loss.py -d market1501 -a resnet50 --combine-trainval --logs-dir examples/logs/softmax-loss/market1501-resnet50``
   ResNet-50 OIM          60.9     37.3         63.6       82.1           ``python examples/oim_loss.py -d market1501 -a resnet50 --combine-trainval --oim-scalar 20 --logs-dir examples/logs/oim-loss/market1501-resnet50``
   ========= ============ ======== ============ ========== ============== ===============

.. _dukemtmc-benchmark:

^^^^^^^^
DukeMTMC
^^^^^^^^

   ========= ============ ======== ============ ========== ============== ===============
   Net       Loss         Mean AP  CMC allshots CMC cuhk03 CMC market1501 Training Script
   ========= ============ ======== ============ ========== ============== ===============
   Inception Triplet      N/A      N/A          N/A        N/A            N/A
   Inception Softmax      34.0     17.4         39.2       54.4           ``python examples/softmax_loss.py -d dukemtmc -a inception --combine-trainval --epochs 70 --logs-dir examples/logs/softmax-loss/dukemtmc-inception``
   Inception OIM          40.6     22.4         45.3       61.7           ``python examples/oim_loss.py -d dukemtmc -a inception --combine-trainval --oim-scalar 30 --epochs 70 --logs-dir examples/logs/oim-loss/dukemtmc-inception``
   ResNet-50 Triplet      **54.6** **34.6**     **57.5**   **73.1**       ``python examples/triplet_loss.py -d dukemtmc -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/dukemtmc-resnet50``
   ResNet-50 Softmax      40.7     23.7         44.3       62.5           ``python examples/softmax_loss.py -d dukemtmc -a resnet50 --combine-trainval --logs-dir examples/logs/softmax-loss/dukemtmc-resnet50``
   ResNet-50 OIM          47.4     29.2         50.4       68.1           ``python examples/oim_loss.py -d dukemtmc -a resnet50 --combine-trainval --oim-scalar 30 --logs-dir examples/logs/oim-loss/dukemtmc-resnet50``
   ========= ============ ======== ============ ========== ============== ===============

.. ATTENTION::
   No test-time augmentation is used. We have fixed a bug in the learning rate
   scheduler for the Triplet loss. Now the result of ResNet-50 with Triplet loss
   is slightly better than the "original" setting in [hermans2017in]_ (Table 4).
