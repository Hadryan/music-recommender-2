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

from recpy.metrics import precision, recall, map, ndcg, rr


def evaluate_metric(test, test_users, recommended_items, metric, at):
    # evaluate the ranking quality
    metric_ = 0.0
    n_eval = 0
    for test_user in test_users:
        relevant_items = test[test_user].indices
        # evaluate the recommendation list with ranking metrics ONLY
        if metric == ndcg:
            metric_ += ndcg(recommended_items[n_eval], relevant_items, relevance=test[test_user].data, at=at)
        else:
            metric_ += metric(recommended_items[n_eval], relevant_items, at)
        n_eval += 1
    metric_ /= n_eval

    return metric_


def evaluate_metrics(test, test_users, recommended_items, at):

    # evaluate the ranking quality
    precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0
    n_eval = 0
    for test_user in test_users:
        relevant_items = test[test_user].indices
        precision_ += precision(recommended_items[n_eval], relevant_items, at=at)
        recall_ += recall(recommended_items[n_eval], relevant_items, at=at)
        map_ += map(recommended_items[n_eval], relevant_items, at=at)
        mrr_ += rr(recommended_items[n_eval], relevant_items, at=at)
        ndcg_ += ndcg(recommended_items[n_eval], relevant_items, relevance=test[test_user].data, at=at)
        n_eval += 1
    precision_ /= n_eval
    recall_ /= n_eval
    map_ /= n_eval
    mrr_ /= n_eval
    ndcg_ /= n_eval

    return precision_, recall_, map_, mrr_, ndcg_