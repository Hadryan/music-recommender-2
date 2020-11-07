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
import numpy as np
from .base import Recommender

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")


class iALS(Recommender):
    '''
    Implicit Alternating Least Squares model (or Weighed Regularized Matrix Factorization)
    Reference: Collaborative Filtering for Implicit Feedback Datasets (Hu et al., 2008)

    Factorization model for implicit feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    '''

    # TODO: Add support for multiple confidence scaling functions (e.g. linear and log scaling)
    def __init__(self,
                 num_factors=50,
                 reg=0.015,
                 iters=10,
                 alpha=40,
                 init_mean=0.0,
                 init_std=0.1,
                 rnd_seed=42):
        '''
        Initialize the model
        :param item: determines if it is item-based or user-based
        :param num_factors: number of latent factors
        :param reg: regularization term
        :param alpha: scaling factor to compute confidence scores
        :param iters: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param rnd_seed: random seed
        '''

        super(iALS, self).__init__()
        self.num_factors = num_factors
        self.reg = reg
        self.iters = iters
        self.alpha = alpha
        self.init_mean = init_mean
        self.init_std = init_std
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "WRMF-iALS(num_factors={},  reg={}, iters={}, alpha={}, init_mean={}, " \
            "init_std={}, rnd_seed={})".format(
                self.num_factors, self.reg, self.iters, self.alpha, self.init_mean, self.init_std,
                self.rnd_seed
            )

    def fit(self, R, rec):
        self.dataset = R
        self.rec = rec

        # compute the confidence matrix
        C = R.copy().tocsr()
        # use linear scaling here
        # TODO: add log-scaling
        C.data = 1 + self.alpha * C.data
        Ct = C.T.tocsr()
        M, N = R.shape

        # set the seed
        np.random.seed(self.rnd_seed)

        # initialize the latent factors
        self.X = np.random.normal(self.init_mean, self.init_std, size=(M, self.num_factors))
        self.Y = np.random.normal(self.init_mean, self.init_std, size=(N, self.num_factors))

        for it in range(self.iters):
            self.X = self._lsq_solver_fast(C, self.X, self.Y, self.reg)
            self.Y = self._lsq_solver_fast(Ct, self.Y, self.X, self.reg)
            logger.debug('Finished iter {}'.format(it + 1))

    def get_scores(self, user_id):  # multiple user
        # compute the scores using the dot product
        return np.dot(self.X[user_id], self.Y.T)

    def recommend(self, user_id, n):  # multiple user
        # compute the scores
        scores = self.get_scores(user_id)
        nusers = user_id.shape[0]
        recs = -1 * np.ones((nusers, n), dtype=np.int32)  # -1 is an invalid id
        for i in range(nusers):
            # rank items
            ranking = scores[i].argsort()[::-1]
            # remove unrecommendable items
            ranking = self._filter_unrec(ranking)
            # remove items the user already interacted with
            ranking = self._filter_seen(user_id[i], ranking)
            # group all the rankings in the same data structure
            nrecs = min(ranking.shape[0], n)
            recs[i, :nrecs] = ranking[:nrecs]
        return recs

    def _lsq_solver_fast(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            start, end = C.indptr[i], C.indptr[i + 1]
            j = C.indices[start:end]  # indices of the non-zeros in Ci
            ci = C.data[start:end]  # non-zeros in Ci

            Yj = Y[j]  # only the factors with non-zero confidence
            # compute Yt(Ci-I)Y
            aux = np.dot(Yj.T, np.diag(ci - 1.0))
            A += np.dot(aux, Yj)
            # compute YtCi
            b = np.dot(Yj.T, ci)

            X[i] = np.linalg.solve(A, b)
        return X
