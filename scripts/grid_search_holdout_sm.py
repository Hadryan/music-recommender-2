# Copyright (c) 2020 Matthew Rossi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
import pandas as pd
import numpy as np
from datetime import datetime as dt

from recpy.utils.data_utils import read_profile, read_dataset, profile_to_ucm, profile_to_icm, df_to_csr
from recpy.utils.split import holdout

from recpy.recommenders.non_personalized import TopPop, GlobalEffects
from recpy.recommenders.knn import KNNRecommender
from recpy.recommenders.slim import SLIM, MultiThreadSLIM
from recpy.recommenders.fs_slim import fsSLIM, fsMultiThreadSLIM
#from recpy.recommenders.sslim import SSLIM
from recpy.recommenders.ials import iALS
from recpy.recommenders.bpr_fm import BPR_FM
from recpy.recommenders.warp_fm import WARP_FM
from recpy.recommenders.kos_warp_fm import kOS_WARP_FM

from recpy.recommenders.model_merger import ModelMerger
from recpy.recommenders.score_merger_fast import ScoreMerger

from recpy.utils.tuning import grid_search_holdout_sm
from recpy.metrics import precision, recall, map, ndcg, rr
from recpy.utils.eval import evaluate_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


holdout_perc = 0.8
rnd_seed = 1234

metric = map
at = 5

# read input files
logger.info('Reading data/playlists_final.csv')
users, user_to_idx, idx_to_user = read_profile(
    'data/playlists_final.csv',
    key='playlist_id',
    header=0,
    sep='\t'
)
logger.info('Reading data/tracks_final.csv.csv')
items, item_to_idx, idx_to_item = read_profile(
    'data/tracks_final.csv',
    key='track_id',
    header=0,
    sep='\t'
)
logger.info('Reading data/train_final.csv')
interactions, _, _ = read_dataset(
    'data/train_final.csv',
    header=0,
    user_key='playlist_id',
    item_key='track_id',
    sep='\t',
    user_to_idx=user_to_idx,
    item_to_idx=item_to_idx
)
logger.info('Reading data/target_tracks.csv')
rec_items = pd.read_csv('data/target_tracks.csv', header=0, sep='\t')
logger.info('Columns: {}'.format(rec_items.columns.values))

nusers, nitems = len(idx_to_user), len(idx_to_item)
logger.info('The dataset has {} users and {} items'.format(nusers, nitems))

ucm = profile_to_ucm(users, user_to_idx)
icm = profile_to_icm(items, item_to_idx)

# compute the holdout split
logger.info('Computing the {:.0f}% holdout split'.format(holdout_perc * 100))
train_df, test_df = holdout(
    interactions,
    rec_items['track_id'],
    user_key='playlist_id',
    item_key='track_id',
    perc=holdout_perc,
    seed=rnd_seed
)
train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
test = df_to_csr(test_df, nrows=nusers, ncols=nitems)

RecommenderClass = ScoreMerger
# KNNRecommender(item=True,content=False,k=2000,shrinkage=100),
# KNNRecommender(item=False,content=False,k=300,shrinkage=5),
# KNNRecommender(item=False,content=True,k=50,shrinkage=10),
# fsSLIM(item=True, content=False, k=3000,shrinkage=0,l1_penalty=1e-06,l2_penalty=0.001),  # MAP@5: 0.0858
# fsSLIM(item=True, content=True, k=3000, shrinkage=0, l1_penalty=1e-07, l2_penalty=0.0001),  # MAP@5: 0.0907
# fsSLIM(item=False, content=False, k=1500, shrinkage=25, l1_penalty=1e-07, l2_penalty=0.001),  # MAP@5: 0.0811
# fsSLIM(item=False, content=True, k=3000, shrinkage=0, l1_penalty=1e-07, l2_penalty=0.001),  # MAP@5: 0.0231
# iALS(alpha=80, init_mean=0, init_std=0.01, iters=50, num_factors=300, reg=0.1, rnd_seed=42),
# BPR_FM(no_components=500, epoch=50, learning_rate=0.01, item_alpha=0.0001, user_alpha=1e-07),
# WARP_FM(no_components=300, epoch=30, learning_rate=0.01, item_alpha=0.0001, user_alpha=1e-06, max_sampled=2000),
# SSLIM(item=True, k=100, l2_penalty=0.0001, l1_penalty=1e-06)
recommenders = [ModelMerger(item=True,
                            recs=[fsSLIM(item=True, content=False, k=3000, shrinkage=0, l1_penalty=1e-06,
                                           l2_penalty=0.001),
                                    fsSLIM(item=True, content=True, k=3000, shrinkage=0, l1_penalty=1e-07,
                                           l2_penalty=0.0001)],
                            weights=[0.75, 0.25]),
                ModelMerger(item=False,
                            recs=[fsSLIM (item=False, content=False, k=1500, shrinkage=25, l1_penalty=1e-07,
                                          l2_penalty=0.001),
                                  fsSLIM(item=False, content=True, k=3000, shrinkage=0, l1_penalty=1e-07,
                                         l2_penalty=0.001)],
                            weights=[0.75, 0.25])
                ]
param_space = {
    'weights': [[0.5, 0.5], [0.55, 0.45], [0.6, 0.4], [0.65, 0.35], [0.7, 0.3], [0.75, 0.25], [0.8, 0.2], [0.85, 0.15],
                [0.9, 0.1], [0.95, 0.05], [1, 0]]
}

# Tune the hyper-parameters with GridSearchHoldout
logger.info('Tuning {} with GridSearchHoldout'.format(RecommenderClass.__name__))
best_config, holdout_score = grid_search_holdout_sm(RecommenderClass,
                                                    train_df,
                                                    nusers,
                                                    nitems,
                                                    rec_items,
                                                    item_to_idx[rec_items['track_id']],
                                                    recommenders,
                                                    param_space,
                                                    user_features=ucm,
                                                    item_features=icm,
                                                    metric=metric,
                                                    at=at,
                                                    perc=holdout_perc,
                                                    user_key='playlist_id',
                                                    item_key='track_id',
                                                    rnd_seed=rnd_seed)
logger.info('Best configuration:')
logger.info(best_config)
logger.info('Holdout score: {:.4f}'.format(holdout_score))
# Evaluate all the metrics over the hold out split
recommender = RecommenderClass(recommenders)

# train the recommender
logger.info('Recommender: {}'.format(recommender))
tic = dt.now()
logger.info('Training started')
recommender.fit(train, item_to_idx[rec_items['track_id']], ucm, icm)
logger.info('Training completed in {}'.format(dt.now() - tic))

# find out the user in the test set
to_test = np.diff(test.indptr) > 0
test_users = np.arange(nusers)[to_test]
# perform the recommendations
recommender.inner_scores(test_users)
recommended_items = recommender.recommend(test_users, at, **best_config)

# evaluate the ranking quality
precision_, recall_, map_, mrr_, ndcg_ = evaluate_metrics(test, test_users, recommended_items, at)

logger.info('Metrics: {:.4f}'.format(precision_, recall_, map_, mrr_, ndcg_))