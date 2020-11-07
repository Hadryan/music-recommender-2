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

import numpy as np
import pandas as pd


def holdout(data, rec, user_key='user_id', item_key='item_id', perc=0.8, seed=1234):
    # force test set to contain only interactions with recommendable items
    # interactions with a recommendable item might fall in the test set
    rec = np.in1d(data[item_key], rec)
    rec_data = data[rec]
    # all interactions with an unrecommendable item fall in the training set
    unrec_data = data[np.invert(rec)]
    # set the random seed
    rng = np.random.RandomState(seed)
    # shuffle data
    nratings = data.shape[0]
    nrec = rec_data.shape[0]
    shuffle_idx = rng.permutation(nrec)
    test_size = int(nratings * (1-perc))
    # split rec_data according to the shuffled index and the holdout size
    train_split = pd.concat([unrec_data, rec_data.iloc[shuffle_idx[test_size:]]])
    test_split = rec_data.iloc[shuffle_idx[:test_size]]
    return train_split, test_split


def k_fold_cv(data, rec, user_key='user_id', item_key='item_id', k=5, seed=1234):
    # force test set to contain only interactions with recommendable items
    # interactions with a recommendable item might fall in the test set
    rec = np.in1d(data[item_key], rec)
    rec_data = data[rec]
    # all interactions with an unrecommendable item fall in the training set
    unrec_data = data[np.invert(rec)]
    # set the random seed
    rng = np.random.RandomState(seed)
    # shuffle data
    nratings = data.shape[0]
    nrec = rec_data.shape[0]
    shuffle_idx = rng.permutation(nrec)
    fold_size = -(-nrec // k)

    for fidx in range(k):
        train_idx = np.concatenate([shuffle_idx[:fidx * fold_size], shuffle_idx[(fidx + 1) * fold_size:]])
        test_idx = shuffle_idx[fidx * fold_size:(fidx + 1) * fold_size]
        train_split = pd.concat([unrec_data, rec_data.iloc[train_idx]])
        test_split = data.iloc[test_idx]
        yield train_split, test_split
