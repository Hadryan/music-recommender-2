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
import scipy.sparse as sps
from .base import Recommender, argsort, check_matrix
from .similarity import Cosine


class KNNRecommender(Recommender):
    """ KNN recommender"""

    def __init__(self, item=True, content=False,  k=50, shrinkage=100):
        super(KNNRecommender, self).__init__()
        self.item = item
        self.content = content
        self.k = k
        self.shrinkage = shrinkage
        self.distance = Cosine(shrinkage=self.shrinkage)

    def __str__(self):
        return "KNN(item={},content={},k={},shrinkage={})".format(
            self.item, self.content, self.k, self.shrinkage)

    def fit(self, X, rec, content=None):

        if not self.content and content is not None:
            raise ValueError("No content info expected")
        if self.content and content is None:
            raise ValueError("Content info expected")
        if self.item and self.content and X.shape[1] != content.shape[0]:
            raise ValueError("URM and ICM don't match")
        if not self.item and self.content and X.shape[0] != content.shape[0]:
            raise ValueError("URM and UCM don't match")

        self.dataset = X
        self.rec = rec

        if content is None:
            if not self.item:
                X = X.T  # csc
        else:
            X = content.T  # csc
        n = X.shape[1]

        self.W_sparse = self.distance.compute(X)  # csr
        self.W_sparse = check_matrix(self.W_sparse, 'csc', dtype=np.float32)  # csc
        for i in np.arange(n):
            idx_sorted = argsort(self.W_sparse[:, i])  # sort by column
            if idx_sorted.shape[0] > self.k:
                not_top_k = idx_sorted[:-self.k]  # index of the items that DON'T BELONG to the top-k similar items
                self.W_sparse[not_top_k, i] = 0.0  # zero-out the not top-k items for the considered column
        self.W_sparse.eliminate_zeros()

        # item_weights = self.distance.compute(X)  # csr
        # item_weights = check_matrix(item_weights, 'csc', dtype=np.float32)  # csc
        # iterate over each column and keep only the top-k similar items
        # values, rows, cols = [], [], []
        # nitems = self.dataset.shape[1]
        # for i in range(nitems):
        #    idx_sorted = argsort(item_weights[:, i])  # sort by column
        #    top_k_idx = idx_sorted[-self.k:]
        #    values.extend(np.sort(item_weights[top_k_idx, i].data))
        #    rows.extend(np.arange(nitems)[top_k_idx])
        #    cols.extend(np.ones(top_k_idx.shape[0]) * i)
        # self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)

    def get_scores(self, user_id):  # multiple user
        # compute the scores using the dot product
        if self.item:
            user_profile = self._get_user_ratings(user_id)
            scores = user_profile.dot(self.W_sparse)  # csr
        else:
            W_sparse = self.W_sparse[:, user_id].T # csr
            dataset = check_matrix(self.dataset, 'csc', dtype=np.float32) # csc
            scores = W_sparse.dot(dataset) # csr
        return scores

    def recommend(self, user_id, n):  # multiple user
        # compute the scores
        scores = self.get_scores(user_id)
        nusers = user_id.shape[0]
        recs = -1 * np.ones((nusers, n), dtype=np.int32)  # -1 is an invalid id
        for i in range(nusers):
            # rank items
            ranking = argsort(scores[i])[::-1]
            # remove unrecommendable items
            ranking = self._filter_unrec(ranking)
            # remove items the user already interacted with
            ranking = self._filter_seen(user_id[i], ranking)
            # group all the rankings in the same data structure
            nrecs = min(ranking.shape[0], n)
            recs[i, :nrecs] = ranking[:nrecs]
        return recs
