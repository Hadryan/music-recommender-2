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
from .base import Recommender, argsort


class ScoreMerger(Recommender):
    """ Recommendation-based hybrid recommender"""

    def __init__(self, recs, weights):
        super(ScoreMerger, self).__init__()

        # if len(recs) < 2:
        #     raise ValueError("No point in doing this")
        if len(recs) != len(weights):
            raise ValueError("Parameters length don't match")

        self.recs = recs
        self.weights = weights

    def __str__(self):
        str = "Merge these recommeders model: "
        for rec in self.recs:
            str += rec.__str__() + ", "
        return str

    def fit(self, X, rec, user_features=None, item_features=None):
        self.dataset = X
        self.rec = rec

        # go through the list of recommenders and fit them
        for i in range(len(self.recs)):
            if hasattr(self.recs[i], 'content'):
                if not self.recs[i].content:
                    print("KNN or fsSLIM no content")
                    self.recs[i].fit(X, rec)
                else:
                    if not self.recs[i].item:
                        print("KNN or fsSLIM user content")
                        self.recs[i].fit(X, rec, user_features)
                    else:
                        print("KNN or fsSLIM item content")
                        self.recs[i].fit(X, rec, item_features)
            else:
                if hasattr(self.recs[i], 'item'):
                    if not self.recs[i].item:
                        print("SSLIM or ModelMerger user content")
                        self.recs[i].fit(X, rec, user_features)
                    else:
                        print("SSLIM or ModelMerger item content")
                        self.recs[i].fit(X, rec, item_features)
                else:
                    if hasattr(self.recs[i], 'num_factors'):
                        print("iALS")
                        self.recs[i].fit(X, rec)
                    else:
                        print("FM")
                        self.recs[i].fit(X, rec, user_features, item_features)

    def recommend(self, user_id, n):  # multiple user

        # for i in range(len(self.recs)):
        #     sps.save_npz("main_scores{}.npz".format(i), self.recs[i].get_scores(user_id))

        # go through the list of recommenders and get for each their own recommendations
        list_of_scores = []
        for rec in self.recs:
            list_of_scores.append(rec.get_scores(user_id))

        # # go through the list of recommenders and get for each their own recommendations
        # list_of_scores = []
        # for i in range(len(self.recs)):
        #     list_of_scores.append(sps.load_npz("main_scores{}.npz".format(i)))

        # compute the scores using the a weighted avg
        scores = self.weights[0] * list_of_scores[0]
        for i in range(len(self.recs) - 1):
            scores += self.weights[i+1] * list_of_scores[i+1]

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