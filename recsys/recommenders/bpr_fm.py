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
from lightfm import LightFM
from .base import Recommender


class BPR_FM(Recommender):
    """ A factorization machine hybrid recommender with bpr loss"""

    def __init__(self, no_components, epoch, learning_rate, item_alpha, user_alpha):

        """
        Initialize the model
        :param no_components: the dimensionality of the feature latent embeddings.
        :param epoch: the number of epochs used to fit the model.
        :param learning_rate: initial learning rate for the adagrad learning schedule.
        :param item_alpha: L2 penalty on item features.
        :param user_alpha: L1 penalty on user features
        """

        super(BPR_FM, self).__init__()
        self.no_components = no_components
        self.epoch = epoch
        self.learning_schedule = 'adagrad'
        self.loss = 'bpr'
        self.learning_rate = learning_rate
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.model = LightFM(no_components=self.no_components,
                             learning_schedule=self.learning_schedule,
                             loss=self.loss,
                             learning_rate=self.learning_rate,
                             item_alpha=self.item_alpha,
                             user_alpha=self.user_alpha,
                             random_state=1234)

    def __str__(self):
        return "BPR_FM(no_components={},learning_schedule={},loss={},learning_rate={},item_alpha={},user_alpha={})".format(
            self.no_components, self.learning_schedule, self.loss, self.learning_rate, self.item_alpha, self.user_alpha)

    def fit(self, train, rec, user_features=None, item_features=None):
        self.dataset = train
        self.rec = rec
        self.user_features = sps.hstack((sps.identity(user_features.shape[0], format='csr'), user_features))
        self.item_features = sps.hstack((sps.identity(item_features.shape[0], format='csr'), item_features))
        self.model.fit(train,
                       user_features=self.user_features,
                       item_features=self.item_features,
                       epochs=self.epoch)

    def get_scores(self, user_id):  # multiple user
        no_user_id = user_id.shape[0]
        no_item_id = self.rec.shape[0]

        # Adapt my input interface to the lightFM one
        user_ids = np.repeat(user_id, no_item_id * np.ones(no_user_id, dtype=np.int64))
        item_ids = np.tile(self.rec.values, no_user_id)

        predictions = self.model.predict(user_ids,
                                         item_ids,
                                         user_features=self.user_features,
                                         item_features=self.item_features)

        # Adapt lightfm.model.predict output to my code
        return np.reshape(predictions, (no_user_id,no_item_id))

    def recommend(self, user_id, n):  # multiple user
        no_user_id = user_id.shape[0]
        scores = self.get_scores(user_id)
        recs = -1 * np.ones((no_user_id, n), dtype=np.int32)  # -1 is an invalid id
        for i in range(no_user_id):
            # rank items
            ranking = self.rec.values[np.argsort(scores[i])[::-1]]
            # remove items the user already interacted with
            ranking = self._filter_seen(user_id[i], ranking)
            # group all the rankings in the same data structure
            nrecs = min(ranking.shape[0], n)
            recs[i, :nrecs] = ranking[:nrecs]
        return recs
