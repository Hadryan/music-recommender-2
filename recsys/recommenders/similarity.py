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
from .base import check_matrix


class ISimilarity(object):
    """Abstract interface for the similarity metrics"""

    def __init__(self, shrinkage=10):
        self.shrinkage = shrinkage

    def compute(self, X):
        pass


class Cosine(ISimilarity):
    def compute(self, X):
        # convert to csc matrix for faster column-wise operations
        X = check_matrix(X, 'csc', dtype=np.float32)

        # 1) normalize the columns in X
        # compute the column-wise norm
        # NOTE: this is slightly inefficient. We must copy X to compute the column norms.
        # A faster solution is to  normalize the matrix inplace with a Cython function.
        Xsq = X.copy()
        Xsq.data **= 2
        norm = np.sqrt(Xsq.sum(axis=0))
        norm = np.asarray(norm).ravel()
        # norm += 1e-6
        # # compute the number of non-zeros in each column
        # # NOTE: this works only if X is instance of sparse.csc_matrix
        # col_nnz = np.diff(X.indptr)
        # # then normalize the values in each column
        # X.data /= np.repeat(norm, col_nnz)
        #
        # # 2) compute the cosine similarity using the dot-product
        dist = X.T.dot(X)  # csr
        # if self.shrinkage > 0:
        #     dist = self.apply_shrinkage(X, dist)
        # # zero out diagonal values
        # dist.setdiag(np.zeros(X.shape[1]))
        # # remove the zero elements
        # dist.eliminate_zeros()

        n = X.shape[1]

        for i in np.arange(len(dist.indptr) - 1):
            if dist.indptr[i] != dist.indptr[i + 1]:  # happens due to how isim is computed
                dist.data[dist.indptr[i]:dist.indptr[i + 1]] = dist.data[dist.indptr[i]:dist.indptr[i + 1]] / (
                    norm[dist.indices[dist.indptr[i]:dist.indptr[i + 1]]] * norm[i] + self.shrinkage)
        dist.setdiag(np.zeros(n))
        dist.eliminate_zeros()

        return dist

    def apply_shrinkage(self, X, dist):  # TODO: do it in a memory efficient way
        # create an "indicator" version of X (i.e. replace values in X with ones)
        X_ind = X.copy()
        X_ind.data = np.ones_like(X_ind.data)
        # compute the co-rated counts
        co_counts = X_ind.T.dot(X_ind)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        dist.data *= co_counts.data / (co_counts.data + self.shrinkage)
        return dist
