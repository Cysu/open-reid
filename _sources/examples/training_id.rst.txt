============================
Training Identification Nets
============================

This example will present how to train nets with identification loss on popular
datasets.

The objective of training an identification net is to learn good feature
representation for persons. If the features of the same person are similar,
while the features of different people are dissimilar, then querying a target
person from a gallery database would become easy.

Different loss functions could be adopted for this purpose, for example,

- Softmax cross entropy loss [zheng2016person]_ [xiao2016learning]_
- Triplet loss [hermans2017in]_
- Online instance matching (OIM) loss [xiaoli2017joint]_


.. _head-first-example:

------------------
Head First Example
------------------

After cloning the repository, we can start with training an Inception net on
VIPeR from scratch

.. code-block:: shell

   python examples/inception.py -d viper -b 64 -j 2 --loss xentropy --logs-dir logs/inception-viper-xentropy

This script automatically downloads the VIPeR dataset and starts training, with
batch size of 64 and two processes for data loading. Softmax cross entropy is
used as the loss function. The training log should be print to screen as well as
saved to ``logs/inception-viper-xentropy/log.txt``. When training ends, it will
evaluate the best model (the one with best validation performance) on the test
set, and report several commonly used metrics.


.. _training-options:

----------------
Training Options
----------------

Many training options are available through command line arguments. See all the
options by ``python examples/inception.py -h``. Here we elaborate on several
commonly used options.

.. _data-options:

^^^^^^^^
Datasets
^^^^^^^^

Specify the dataset by ``-d name``, where ``name`` can be one of ``cuhk03``,
``market1501``, ``dukemtmc``, and ``viper`` currently. For some datasets that
cannot be downloaded automatically, running the script will raise an error with
a link to the dataset. One may need to manually download it and put it to the
directory instructed also by the error message.

.. _gpu-options:

^^^^^^^^^^^^^^^^^^^^^^^^
Multi-GPU and Batch Size
^^^^^^^^^^^^^^^^^^^^^^^^

All the examples support data parallel training on multiple GPUs. By default,
the program will use all the GPUs listed in ``nvidia-smi``. To control which
GPUs to be used, one need to specify the environment variable
``CUDA_VISIBLE_DEVICES`` before running the python script. For example,

.. code-block:: shell

   # 4 GPUs, with effective batch size of 256
   CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/inception.py -d viper -b 256 --lr 0.1

   # 1 GPU, reduce the batch size to 64, lr to 0.025
   CUDA_VISIBLE_DEVICES=0 python examples/inception.py -d viper -b 64 --lr 0.025

Note that the effective batch size specified by the ``-b`` option will be
divided automatically by the number of GPUs. For example, 4 GPUs with ``-b 256``
will have 64 minibatch samples on each GPU.

In the second command above, we reduce the batch size and initial learning rate
to 1/4, in order to adapt the original 4 GPUs setting to only 1 GPU.

.. _resume-options:

^^^^^^^^^^^^^^^^^^^^^^^
Resume from Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^

After each training epoch, the script would save a latest ``checkpoint.pth.tar``
in the specified logs directory, and update a ``model_best.pth.tar`` if the
model achieves the best validation performance so far. To resume from this
checkpoint, just run the script with ``--resume /path/to/checkpoint.pth.tar``.

.. _eval-options:

^^^^^^^^^^^^^^^^^^^^^^^^
Evaluate a Trained Model
^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate a trained model, just run the script with ``--resume
/path/to/model_best.pth.tar --evaluate``. Different evaluation metrics,
especially different versions of CMC could lead to drastically different
numbers.


.. _tips-and-tricks:

---------------
Tips and Tricks
---------------

Training a baseline network can be tricky. Many options and parameters could
(significantly) affect the reported performance number. Here we list some tips
and tricks for experiments.

Combine train and val
   One can first use separate training and validation set to tune the
   hyperparameters, then fix the hyperparameters and combine both sets togehter
   to train a final model. This can be done by appending an option
   ``--combine-trainval``, and could lead to much better performance on the
   test set.

Input size
   Larger input image size could benefit the performance. But it depends on the
   network architecture.

Multi-scale multi-crop test
   Using multi-scale multi-crop for test normally guarantees performance gain.
   However, it sacrifies the running speed significantly. We have not
   implemented this yet.

Classifier initialization for softmax cross entropy loss
   We found that initializing the softmax classifier weight with normal
   distribution ``std=0.001`` generally leads to better performance. It is also
   important to use larger learning rate for the classifier if underlying CNN is
   already pretrained.


----------
References
----------

.. [zheng2016person] L. Zheng, Y. Yang, and A.G. Hauptmann. Person Re-identification: Past, Present and Future. *arXiv:1610.02984*, 2014.
.. [xiao2016learning] T. Xiao, H. Li, W. Ouyang, and X. Wang. Learning deep feature representations with domain guided dropout for person re-identification. In *CVPR*, 2016.
.. [xiaoli2017joint] T. Xiao\*, S. Li\*, B. Wang, L. Lin, and X. Wang. Joint Detection and Identification Feature Learning for Person Search. In *CVPR*, 2017.
.. [hermans2017in] A. Hermans, L. Beyer, and B. Leibe. In Defense of the Triplet Loss for Person Re-Identification. *arXiv:1703.07737*, 2017.