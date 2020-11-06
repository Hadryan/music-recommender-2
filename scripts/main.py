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
from recpy.recommenders.score_merger import ScoreMerger


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

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
logger.info('Reading data/target_playlists.csv')
targets = pd.read_csv('data/target_playlists.csv', header=0, sep='\t')
logger.info('Columns: {}'.format(targets.columns.values))

nusers, nitems, ntargets = len(idx_to_user), len(idx_to_item), len(targets)

# create the user content matrix
ucm = profile_to_ucm(users, user_to_idx)
# create the item content matrix
icm = profile_to_icm(items, item_to_idx)
# create the user rating matrix
urm = df_to_csr(interactions, nrows=nusers, ncols=nitems)




# train the recommender
recommender = ModelMerger(item=True,
                          recs=[fsSLIM(item=True, content=False, k=3000, shrinkage=0, l1_penalty=1e-06, l2_penalty=0.001),
                                fsSLIM(item=True, content=True, k=3000, shrinkage=0, l1_penalty=1e-07, l2_penalty=0.0001)],
                          weights=[0.75, 0.25])

logger.info('Recommender: {}'.format(recommender))
tic = dt.now()
logger.info('Training started')
recommender.fit(urm, item_to_idx[rec_items['track_id']], icm)
logger.info('Training completed in {}'.format(dt.now() - tic))

rec_length = 5

# open the prediction file
pfile = open('submissions/submission.csv', 'w')
header = 'playlist_id,track_ids' + '\n'
pfile.write(header)

# find out the user in the test set
target_users = targets['playlist_id'].values
# for some reason the following doesn't work in python 2.7
# target_idx = user_to_idx[target_users].data
target_idx = user_to_idx[target_users].values
# perform recommendations with the main technique
recommended_items = recommender.recommend(target_idx, rec_length)

# train the secondary recommender
secondary = ModelMerger(item=False,
                        recs=[fsSLIM(item=False, content=False, k=1500, shrinkage=25, l1_penalty=1e-07, l2_penalty=0.001),
                              fsSLIM(item=False, content=True, k=3000, shrinkage=0, l1_penalty=1e-07, l2_penalty=0.001)],
                        weights=[0.75, 0.25])

logger.info('Recommender: {}'.format(secondary))
tic = dt.now()
logger.info('Training started')
secondary.fit(urm, item_to_idx[rec_items['track_id']], ucm)
logger.info('Training completed in {}'.format(dt.now() - tic))

# fill the recommendations with the secondary one
for i in range(ntargets):
    nrec = recommended_items[i].argmin()
    if recommended_items[i, nrec] == -1:
        filler = secondary.recommend(target_idx[i], rec_length)
        usable = np.in1d(filler, recommended_items[i], assume_unique=True, invert=True)
        recommended_items[i, nrec:] = filler[usable][:(rec_length-nrec)]

# write the recommendations into the file
for i in range(ntargets):
    # write the recommendation list to file, one user per line
    s = str(target_users[i]) + ','
    s += ' '.join([str(idx_to_item[x]) for x in recommended_items[i]]) + '\n'
    pfile.write(s)

# close the prediction file
pfile.close()
logger.info('Recommendations written to submission.csv')
