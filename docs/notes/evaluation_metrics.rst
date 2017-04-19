.. _evaluation-metrics:

==================
Evaluation Metrics
==================

Cumulative Matching Characteristics (CMC) curves are the most popular evaluation
metrics for person re-identification methods. Consider a simple
*single-gallery-shot* setting, where each gallery identity has only one
instance. For each query, an algorithm will rank all the gallery samples
according to their distances to the query from small to large, and the CMC top-k
accuracy is

.. math::
   Acc_k = \begin{cases}
      1 & \text{if top-$k$ ranked gallery samples contain the query identity} \\
      0 & \text{otherwise}
   \end{cases},

which is a shifted `step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_.
The final CMC curve is computed by averaging the shifted step functions over all
the queries.

While the *single-gallery-shot* CMC is well defined, it does not have a common
agreement when it comes to the *multi-gallery-shot* setting, where each gallery
identity could have multiple instances. For example,
`CUHK03 <www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf>`_ and
`Market-1501 <http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf>`_
calculated the CMC curves and CMC top-k accuracy quite differently. To be
specific,

- `CUHK03 <www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf>`_:
  Query and gallery sets are from different camera views. For each query, they
  randomly sample one instance for each gallery identity, and compute a CMC
  curve in the *single-gallery-shot* setting. The random sampling is repeated
  for :math:`N` times and the expected CMC curve is reported.

- `Market-1501 <http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf>`_:
  Query and gallery sets could have same camera views, but for each individual
  query identity, his/her gallery samples from the same camera are excluded.
  They do not randomly sample only one instance for each gallery identity. This
  means the query will always match the "easiest" positive sample in the gallery
  while does not care other harder positive samples when computing CMC.
