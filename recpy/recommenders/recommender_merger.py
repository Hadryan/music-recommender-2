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

import random as rnd
import numpy as np
from .base import Recommender, argsort, check_matrix


class RecommenderMerger(Recommender):
    """ Recommendation-based hybrid recommender"""

    def __init__(self, recs, probabilities):
        super(RecommenderMerger, self).__init__()

        if len(recs) < 2:
            raise ValueError("No point in doing this")
        if len(recs) != len(probabilities):
            raise ValueError("Parameters length don't match")
        if sum(probabilities) != 1.0:
            raise ValueError("The probabilities sum should be 1")

        self.recs = recs
        self.probabilities = probabilities

    def __str__(self):
        str = "Merge these recommeders model: "
        for rec in self.recs:
            str += rec.__str__() + ", "
        return str

    def fit(self, X, rec, user_features=None, item_features=None):  # TODO: try to change it as model merger
        self.dataset = X
        self.rec = rec

        # go through the list of recommenders and fit them
        for i in range(len(self.recs)):
            if hasattr(self.recs[i], 'content'):
                if not self.recs[i].content:
                    self.recs[i].fit(X, rec)
                else:
                    if not self.recs[i].item:
                        self.recs[i].fit(X, rec, user_features)
                    else:
                        self.recs[i].fit(X, rec, item_features)
            else:
                if hasattr(self.recs[i], 'item'):
                    if not self.recs[i].item:
                        self.recs[i].fit(X, rec, user_features)
                    else:
                        self.recs[i].fit(X, rec, item_features)
                else:
                    if hasattr(self.recs[i], 'num_factors'):
                        self.recs[i].fit(X, rec)
                    else:
                        self.recs[i].fit(X, rec, user_features, item_features)

    def recommend(self, user_id, n):  # multiple user

        rnd.seed(1234)

        nrecs = len(self.recs)

        # go through the list of recommenders and get for each their own recommendations
        list_of_recs = []
        for rec in self.recs:
            list_of_recs.append(rec.recommend(user_id, n))

        nusers = user_id.shape[0]
        recs = -1 * np.ones((nusers, n), dtype=np.int32)  # -1 is an invalid id

        for user in range(nusers):
            cumsum = np.cumsum(self.probabilities)

            # redistribute probability if a list is empty
            for i in range(nrecs):
                if list_of_recs[i][user, 0] == -1:
                    cumsum = self._redistribute_probs(cumsum, i)

            # all lists are empty so, skip to the next user
            if cumsum[0] == 1 and list_of_recs[0][user, 0] == -1:
                continue

            # extract from these list of recommendations the items of the final recommendation
            for rank in range(n):
                # choose the list from which to extract the item
                l = rnd.random()
                for k in range(len(cumsum)):
                    if l < cumsum[k]:
                        break
                else:
                    raise ValueError("Something wrong within the code!")

                # starting from the top of the n-elements list extract the item taking into account repetition problems
                recs[user, rank] = list_of_recs[k][user, 0]

                # remove the selected item from the other lists
                for i in range(nrecs):
                    pos = np.where(list_of_recs[i][user, :] == recs[user, rank])[0]
                    if len(pos) == 1:
                        list_of_recs[i][user, :] = np.hstack((list_of_recs[i][user, :pos[0]], np.roll(list_of_recs[i][user, pos[0]:], -1)))
                        list_of_recs[i][user, n-1] = -1

                # redistribute probability if all the elements of a list have been already used
                for i in range(nrecs):
                    if list_of_recs[i][user, 0] == -1:
                        cumsum = self._redistribute_probs(cumsum, i)

                # all lists are empty so, skip to the next user
                if cumsum[0] == 1 and list_of_recs[0][user, 0] == -1:
                    break

        return recs

    def _redistribute_probs(self, cumsum, pos):
        probs = np.hstack((cumsum[0], np.diff(cumsum)))

        if probs[pos] == 1:
            return np.ones_like(cumsum)

        redistrib_factor = probs[pos] / (1 - probs[pos])
        for i in range(len(cumsum)):
            if i != pos:
                probs[i] = probs[i] + redistrib_factor * probs[i]
            else:
                probs[i] = 0

        cumsum = np.cumsum(probs)

        if sum(cumsum) == 0:
            return np.ones_like(cumsum)

        return cumsum

