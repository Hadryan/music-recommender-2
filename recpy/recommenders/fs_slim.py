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
from sklearn.linear_model import ElasticNet

from multiprocessing import Pool
from functools import partial

# TODO: fix the fsMultiThreadSLIM

class fsSLIM(Recommender):
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
                 content=False,
                 k=50,
                 shrinkage=100,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True):
        super(fsSLIM, self).__init__()
        self.item = item
        self.content = content
        self.k = k
        self.shrinkage = shrinkage
        self.distance = Cosine(shrinkage=self.shrinkage)
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.alpha = self.l1_penalty + self.l2_penalty
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)

    def __str__(self):
        return "fsSLIM (item={},k={},shrinkage={},l1_penalty={},l2_penalty={},positive_only={})".format(
            self.item,self.k, self.shrinkage, self.l1_penalty, self.l2_penalty, self.positive_only
        )

    def _fs(self, sim, j):
        fs = argsort(sim[j])[-self.k:]
        # if len(fs) < self.k:
        # fs = np.arange(sim.shape[0])
        return fs

    def fit(self, X, rec, content=None):

        if self.content is False and content is not None:
            raise ValueError("No content info expected")
        if self.content is True and content is None:
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
            X = content.T
        n = X.shape[1]

        X = check_matrix(X, 'csc', dtype=np.float32)

        # initialize the itemkNN model
        sim = self.distance.compute(X)  # csr

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=self.alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False)

        # we'll store the W matrix into a sparse csr_matrix
        # let's initialize the vectors used by the sparse.csc_matrix constructor
        values, rows, cols = [], [], []

        # fit each item's factors sequentially (not in parallel)
        for j in range(n):
            # select the most important feature with kNN
            fs = self._fs(sim, j)  # sim is csr -> done row-wise, as we wanted

            # checks if at least one feature has been selected
            if fs.shape[0]:
                # get the target column
                y = X[:, j].toarray()

                # fit one ElasticNet model per column
                self.model.fit(X[:, fs], y)

                # self.model.coef_ contains the coefficient of the ElasticNet model
                # let's keep only the non-zero values

                # might work better using self.model.
                if self.model.coef_.shape:
                    nnz_idx = self.model.coef_ > 0.0
                    values.extend(self.model.coef_[nnz_idx])
                    rows.extend(fs[nnz_idx])
                    cols.extend(np.ones(nnz_idx.sum()) * j)
                else:
                    if self.model.coef_ > 0.0:
                        values.extend([self.model.coef_])
                        rows.extend(fs)
                        cols.extend([j])

        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float32)

    def get_scores(self, user_id):  # multiple users
        # compute the scores using the dot product
        if self.item:
            user_profile = self._get_user_ratings(user_id)
            scores = user_profile.dot(self.W_sparse)  # csr
        else:
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


class fsMultiThreadSLIM(fsSLIM):
    def __init__(self,
                 item=True,
                 k=50,
                 shrinkage=100,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True,
                 workers=4):
        super(fsMultiThreadSLIM, self).__init__(item=item,
                                                k=k,
                                                shrinkage=shrinkage,
                                                l1_penalty=l1_penalty,
                                                l2_penalty=l2_penalty,
                                                positive_only=positive_only)
        self.workers = workers

    def __str__(self):
        return "fsSLIM_mt (item={},k={},shrinkage={},l1_penalty={},l2_penalty={},positive_only={},workers={})".format(
            self.item,self.k, self.shrinkage, self.l1_penalty, self.l2_penalty, self.positive_only, self.workers
        )

    def _partial_fit(self, j, X, sim):
        model = ElasticNet(alpha=self.alpha,
                           l1_ratio=self.l1_ratio,
                           positive=self.positive_only,
                           fit_intercept=False,
                           copy_X=False)
        # select the most important feauture with kNN
        fs = self._fs(sim, j)
        # checks if at least one feature has been selected
        if fs.shape[0]:
            # get the target column
            y = X[:, j].toarray()
            # fit one ElasticNet model per column
            model.fit(X[:, fs], y)
            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            if self.model.coef_.shape:
                nnz_idx = model.coef_ > 0.0
                values = model.coef_[nnz_idx]
                rows = fs[nnz_idx]
                cols = np.ones(nnz_idx.sum()) * j
                return values, rows, cols
            else:
                if self.model.coef_ > 0.0:
                    values = [self.model.coef_]
                    rows = fs
                    cols = [j]
                    return values, rows, cols
        return [], [], []

    def fit(self, X, rec, content=None):
        self.dataset = X
        self.rec = rec

        if content is None:
            if not self.item:
                X = X.T  # csc
        else:
            X = content.T  # csc
        n = X.shape[1]

        X = check_matrix(X, 'csc', dtype=np.float32)

        # initialize the itemkNN model
        sim = self.distance.compute(X)  # csr

        # fit item's factors in parallel
        _pfit = partial(self._partial_fit, X=X, sim=sim)
        pool = Pool(processes=self.workers)
        res = pool.map(_pfit, np.arange(n))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)
        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float32)
