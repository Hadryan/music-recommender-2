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
from .base import Recommender, argsort, check_matrix


class ModelMerger(Recommender):
    """ Model-based hybrid recommender"""

    def __init__(self, item, recs):
        super(ModelMerger, self).__init__()

        if len(recs) < 2:
            raise ValueError("No point in doing this")

        self.item = item
        self.recs = recs

    def __str__(self):
        str = "Merge these recommeders model: "
        for rec in self.recs:
            str += rec.__str__() + ", "
        return str

    def fit(self, X, rec, content=None):
        self.dataset = X
        self.rec = rec

        # go through the list of recommenders and fit them
        for rec in self.recs:
            if not rec.content:
                rec.fit(X,rec)
            else:
                rec.fit(X, rec, content)

    def merge(self, weights):
        if len(self.recs) != len(weights):
            raise ValueError("Parameters length don't match")

        # take their model and average them using the weights
        W_sparse = self.recs[0].W_sparse.tocsc(copy=True)
        W_sparse.data = weights[0] * W_sparse.data
        self.W_sparse = W_sparse
        for i in range(len(self.recs) - 1):
            W_sparse = self.recs[i+1].W_sparse.tocsc(copy=True)
            W_sparse.data = weights[i+1] * W_sparse.data
            self.W_sparse += W_sparse

    def recommend(self, user_id, n):  # multiple user
        # compute the scores using the dot product
        if self.item:
            user_profile = self._get_user_ratings(user_id)
            scores = user_profile.dot(self.W_sparse)  # csr
        else:
            W_sparse = self.W_sparse[:, user_id].T # csr
            dataset = check_matrix(self.dataset, 'csc', dtype=np.float32) # csc
            scores = W_sparse.dot(dataset) # csr

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
