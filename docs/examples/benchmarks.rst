Benchmarks
==========

Benchmarks for different models and loss functions on various datasets.

All the experiments are conducted under the settings of:

* 4 GPUs for training, meaning that ``CUDA_VISIBLE_DEVICES=0,1,2,3`` is set for the training scripts
* Total effective batch size of 256. Consider reducing batch size and learning rate if you only have one GPU. See [] for more details.
* Use the default dataset split ``--split 0``, but combine training and validation sets for training models ``--combine-trainval``
* Use the default random seed ``--seed 1``
* Use Euclidean distance directly for evaluation
* Full set of evaluation metrics are reported. See [] for more explanations of each evaluation metric.

CUHK03
^^^^^^

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

   ========= ============ ============ ========== ============== ================================
   Net       Loss         CMC allshots CMC cuhk03 CMC market1501 Training Parameters
   ========= ============ ============ ========== ============== ================================
   Inception CrossEntropy 13.8         35.1       50.0           ``python examples/inception.py -d duke --loss xentropy``
   Inception OIM          18.4         40.7       56.1           ``python examples/inception.py -d duke --loss oim --oim-scalar 20``
   ResNet-50 CrossEntropy 18.9         38.5       58.1           ``python examples/resnet.py -d duke --loss xentropy``
   ResNet-50 OIM          23.9         44.7       63.3           ``python examples/resnet.py -d duke --loss oim --oim-scalar 30``
   ========= ============ ============ ========== ============== ================================
