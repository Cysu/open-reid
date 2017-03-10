from __future__ import print_function

from .metrics import cmc


def evaluate_cmc(distmat, query, gallery, topk=(1, 5, 10)):
    distmat = distmat.cpu().numpy()

    query_ids = [pid for _, pid, _ in query]
    gallery_ids = [pid for _, pid, _ in gallery]
    query_cams = [cam for _, _, cam in query]
    gallery_cams = [cam for _, _, cam in gallery]

    # Compute both new and old cmc scores
    cmc_configs = {
        'new': dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),
    }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('new', 'cuhk03', 'market1501'))
    for k in topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['new'][k - 1], cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the new cmc top-1 score for validation criterion
    return cmc_scores['new'][0]
