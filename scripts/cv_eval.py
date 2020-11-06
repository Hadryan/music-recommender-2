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
from collections import OrderedDict
from datetime import datetime as dt

import numpy as np
import pandas as pd

from recpy.utils.data_utils import read_profile, read_dataset, profile_to_ucm, profile_to_icm, df_to_csr
from recpy.utils.split import k_fold_cv
from recpy.utils.eval import evaluate_metrics

from recpy.recommenders.non_personalized import TopPop, GlobalEffects
from recpy.recommenders.knn import KNNRecommender
from recpy.recommenders.slim import SLIM, MultiThreadSLIM
from recpy.recommenders.fs_slim import fsSLIM, fsMultiThreadSLIM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('top_pop', TopPop),
    ('global_effects', GlobalEffects),
    ('knn', KNNRecommender),
    ('SLIM', SLIM),
    ('SLIM_mt', MultiThreadSLIM),
    ('fsSLIM', fsSLIM),
    ('fsSLIM_mt', fsMultiThreadSLIM),
])

cv_folds = 5
header = 0
rnd_seed = 1234
recommender = 'fsSLIM'
params = 'k=100,shrinkage=25,l1_penalty=0.00002,l2_penalty=0.01'  # similarity=cosine,sparse_weights=True
rec_length = 5
prediction_file = None

# get the recommender class
assert recommender in available_recommenders, 'Unknown recommender: {}'.format(recommender)
RecommenderClass = available_recommenders[recommender]
# parse recommender parameters
init_args = OrderedDict()
if params:
    for p_str in params.split(','):
        key, value = p_str.split('=')
        try:
            init_args[key] = eval(value)
        except:
            init_args[key] = value

# read input files
logger.info('Reading data/playlists_final.csv')
users, user_to_idx, idx_to_user = read_profile(
    'data/playlists_final.csv',
    key='playlist_id',
    header=0,
    sep='\t'
)
logger.info('Reading data/tracks_final.csv')
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

# evaluate the recommendation quality with k-fold cross-validation
logger.info('Running {}-fold Cross Validation'.format(cv_folds))
precision_, recall_, map_, mrr_, ndcg_ = np.zeros(cv_folds), np.zeros(cv_folds), np.zeros(cv_folds), \
                                         np.zeros(cv_folds), np.zeros(cv_folds)
at = rec_length
nfold = 0
for train_df, test_df in k_fold_cv(interactions,
                                   rec_items['track_id'],
                                   user_key='playlist_id',
                                   item_key='track_id',
                                   k=cv_folds,
                                   seed=rnd_seed):
    logger.info(train_df.shape)
    logger.info(test_df.shape)
    logger.info('Fold {}'.format(nfold + 1))
    train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
    test = df_to_csr(test_df, nrows=nusers, ncols=nitems)

    # train the recommender
    recommender = RecommenderClass(**init_args)
    logger.info('Recommender: {}'.format(recommender))
    tic = dt.now()
    logger.info('Training started')
    recommender.fit(train, item_to_idx[rec_items['track_id']])
    logger.info('Training completed in {}'.format(dt.now() - tic))

    # find out the user in the test set
    to_test = np.diff(test.indptr) > 0
    test_users = np.arange(nusers)[to_test]
    # perform the recommendations
    recommended_items = recommender.recommend(test_users, rec_length)

    # evaluate the ranking quality
    precision_[nfold], recall_[nfold], map_[nfold], mrr_[nfold], ndcg_[nfold] = evaluate_metrics(test,
                                                                                                 test_users,
                                                                                                 recommended_items,
                                                                                                 at)

    nfold += 1

logger.info('Ranking quality')
logger.info('Precision@{}: {:.4f}'.format(at, precision_.mean()))
logger.info('Recall@{}: {:.4f}'.format(at, recall_.mean()))
logger.info('MAP@{}: {:.4f}'.format(at, map_.mean()))
logger.info('MRR@{}: {:.4f}'.format(at, mrr_.mean()))
logger.info('NDCG@{}: {:.4f}'.format(at, ndcg_.mean()))
