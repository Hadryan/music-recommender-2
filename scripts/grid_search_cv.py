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

from recpy.utils.data_utils import read_profile, read_dataset, df_to_csr, profile_to_ucm, profile_to_icm
from recpy.utils.split import holdout

from recpy.recommenders.non_personalized import TopPop, GlobalEffects
from recpy.recommenders.knn import KNNRecommender
from recpy.recommenders.slim import SLIM, MultiThreadSLIM
from recpy.recommenders.fs_slim import fsSLIM, fsMultiThreadSLIM
# from recpy.recommenders.mf import FunkSVD, IALS_numpy, BPRMF

from recpy.utils.tuning import grid_search_cv
from recpy.metrics import precision, recall, map, ndcg, rr
from recpy.utils.eval import evaluate_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


holdout_perc = 0.8
rnd_seed = 1234

metric = map
cv_folds = 5
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

nusers, nitems = len(idx_to_user), len(idx_to_item)
logger.info('The dataset has {} users and {} items'.format(nusers, nitems))

# identify recommendable items
logger.info('Reading data/target_tracks.csv')
rec_items = pd.read_csv('data/target_tracks.csv', header=0, sep='\t')
logger.info('Columns: {}'.format(rec_items.columns.values))

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

# TODO: not all the algorithms support multiple user recommendation

#
# TopPop
# param_space = {}
#
# GlobalEffects
# param_space = {
#     'lambda_user': np.arange(10, 100, 20),
#     'lambda_item': np.arange(10, 100, 20),
# }
#
# ItemKNNRecommender
# param_space = {
#     'k': np.arange(50, 200, 50),
#     'shrinkage': np.arange(0, 100, 20)
# }
#
# SLIM
# param_space = {
#     'l1_penalty': np.logspace(-4, 2, 5),
#     'l2_penalty': np.logspace(-4, 2, 5),
# }
#
# FunkSVD
# param_space = {
#     'num_factors': [20],
#     'iters': [10],
#     'lrate': np.logspace(-4, -1, 5),
#     'reg': np.logspace(-3, 1, 5),
# }
#
# IALS_numpy
# param_space = {
#     'num_factors': [20],
#     'iters': [10],
#     'alpha': np.arange(20, 100, 20),
#     'reg': np.logspace(-3, 1, 5),
# }
#
# BPRMF
# param_space = {
#     'num_factors': [20],
#     'iters': [10],
#     'sample_with_replacement': [True],
#     'lrate': np.logspace(-3, -1, 4),
#     'user_reg': np.logspace(-3, 0, 4),
#     'pos_reg': np.logspace(-3, 0, 4),
#     # 'neg_reg': np.logspace(-4, 0, 5),
# }


RecommenderClass = KNNRecommender
param_space = {
    'k': np.arange(50, 200, 50),
    'shrinkage': np.arange(0, 100, 20)
}
# Tune the hyper-parameters with GridSearchHoldout
logger.info('Tuning {} with GridSearchCV'.format(RecommenderClass.__name__))
best_config, cv_score = grid_search_cv(RecommenderClass,
                                       train_df,
                                       nusers,
                                       nitems,
                                       rec_items,
                                       item_to_idx[rec_items['track_id']],
                                       param_space,
                                       metric=metric,
                                       at=at,
                                       cv_folds=cv_folds,
                                       user_key='playlist_id',
                                       item_key='track_id',
                                       rnd_seed=rnd_seed)
logger.info('Best configuration:')
logger.info(best_config)
logger.info('CV score: {:.4f}'.format(cv_score))
# Evaluate all the metrics over the hold out split
recommender = RecommenderClass(**best_config)

# train the recommender
logger.info('Recommender: {}'.format(recommender))
tic = dt.now()
logger.info('Training started')
print(train.sum())
recommender.fit(train, item_to_idx[rec_items['track_id']])
logger.info('Training completed in {}'.format(dt.now() - tic))

# find out the user in the test set
to_test = np.diff(test.indptr) > 0
test_users = np.arange(nusers)[to_test]
# perform the recommendations
recommended_items = recommender.recommend(test_users, at)

# evaluate the ranking quality
precision_, recall_, map_, mrr_, ndcg_ = evaluate_metrics(test, test_users, recommended_items, at)

logger.info('Metrics: {}'.format(precision_, recall_, map_, mrr_, ndcg_))