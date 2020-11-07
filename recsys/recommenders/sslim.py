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

from mrec.item_similarity.slim import SLIM

import numpy as np
import scipy.sparse as sps
from .base import Recommender, argsort, check_matrix

class SSLIM(Recommender):
    """
    Train a feature selection Sparse Linear Methods (fsSLIM) item similarity model.

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self,
                 item=True,
                 k=50,
                 l1_penalty=0.1,
                 l2_penalty=0.1):
        super(SSLIM, self).__init__()
        self.item = item
        self.k = k
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.model = SLIM(l1_reg=l1_penalty,
                          l2_reg=l2_penalty,
                          fit_intercept=False,
                          ignore_negative_weights=True,#False,  # TODO: test both with true and false
                          num_selected_features=k,
                          model='fs_sgd')

    def __str__(self):
        return "SSLIM (item={},l1_reg={},l2_reg={},num_selected_features={})".format(
            self.item, self.l1_penalty, self.l2_penalty, self.k
        )

    def fit(self, X, rec, content=None):

        if content is None:
            raise ValueError("Content info expected")
        if self.item and X.shape[1] != content.shape[0]:
            raise ValueError("URM and ICM don't match")
        if not self.item and X.shape[0] != content.shape[0]:
            raise ValueError("URM and UCM don't match")

        self.dataset = X
        self.rec = rec

        if not self.item:
            X = X.T  # csc

        self.model.fit(X, content)

    def get_scores(self, user_id):  # multiple users

        self.W_sparse = self.model.similarity_matrix

        # compute the scores using the dot product
        if self.item:
            user_profile = self._get_user_ratings(user_id)
            scores = user_profile.dot(self.W_sparse) # csr
        else:
            print("Users side")
            W_sparse = self.W_sparse[:, user_id].T # csr
            dataset = check_matrix(self.dataset, 'csc', dtype=np.float32) # csc
            scores = W_sparse.dot(dataset) # csr
        return scores

    def recommend(self, user_id, n):  # multiple users

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