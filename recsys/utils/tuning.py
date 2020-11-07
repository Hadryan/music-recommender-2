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
from sklearn.model_selection import ParameterGrid

from recsys.metrics import precision
from recsys.utils.data_utils import df_to_csr
from recsys.utils.split import holdout, k_fold_cv
from recsys.utils.eval import evaluate_metric

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def grid_search_holdout(RecommenderClass, dataset, nusers, nitems, rec, rec_idx, param_space, user_features=None,
                        item_features=None, metric=precision, at=None, perc=0.8, user_key='user_id', item_key='item_id',
                        rnd_seed=1234):
    """
    Find the best hyper-parameters of a recommender algorithm with Grid Search
    """

    tried_conf = []
    results = np.zeros(np.prod([len(v) for v in param_space.values()]), dtype=np.float32)
    space_size = len(results)
    logger.info('Size of the parameter space: {} ({} holdout trials)'.format(space_size, space_size))
    param_grid = ParameterGrid(param_space)
    # compute the holdout split
    train_df, test_df = holdout(dataset,
                                rec,
                                user_key=user_key,
                                item_key=item_key,
                                perc=perc,
                                seed=rnd_seed)
    still_rec = sum(np.in1d(train_df[item_key], rec))


    train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
    test = df_to_csr(test_df, nrows=nusers, ncols=nitems)

    for i, params in enumerate(param_grid):
        logger.info('Iteration {}/{}: {}'.format(i+1, space_size, params))
        tried_conf.append(params)

        # train the recommender
        recommender = RecommenderClass(**params)
        if user_features is not None and item_features is not None:
            recommender.fit(train, rec_idx, user_features, item_features)
        elif user_features is not None:
            recommender.fit(train, rec_idx, user_features)
        elif item_features is not None:
            recommender.fit(train, rec_idx, item_features)
        else:
            recommender.fit(train, rec_idx)

        # find out the user in the test set
        to_test = np.diff(test.indptr) > 0
        test_users = np.arange(nusers)[to_test]
        # perform the recommendations
        recommended_items = recommender.recommend(test_users, at)

        # evaluate the ranking quality
        results[i] = evaluate_metric(test, test_users, recommended_items, metric, at)
        logger.info('Result: {:.4f}'.format(results[i]))
    # return the best configuration
    best = results.argsort()[-1]
    return tried_conf[best], results[best]


def grid_search_holdout_mm(RecommenderClass, dataset, nusers, nitems, rec, rec_idx, item, recommenders, param_space,
                           user_features=None, item_features=None, metric=precision, at=None, perc=0.8,
                           user_key='user_id', item_key='item_id', rnd_seed=1234):
    """
    Find the best hyper-parameters of a recommender algorithm with Grid Search
    """

    tried_conf = []
    results = np.zeros(np.prod([len(v) for v in param_space.values()]), dtype=np.float32)
    space_size = len(results)
    logger.info('Size of the parameter space: {} ({} holdout trials)'.format(space_size, space_size))
    param_grid = ParameterGrid(param_space)
    # compute the holdout split
    train_df, test_df = holdout(dataset,
                                rec,
                                user_key=user_key,
                                item_key=item_key,
                                perc=perc,
                                seed=rnd_seed)

    train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
    test = df_to_csr(test_df, nrows=nusers, ncols=nitems)

    # train the recommender
    recommender = RecommenderClass(item, recommenders)
    if user_features is not None and item_features is not None:
        recommender.fit(train, rec_idx, user_features, item_features)
    elif user_features is not None:
        recommender.fit(train, rec_idx, user_features)
    elif item_features is not None:
        recommender.fit(train, rec_idx, item_features)
    else:
        recommender.fit(train, rec_idx)

    for i, params in enumerate(param_grid):
        logger.info('Iteration {}/{}: {}'.format(i+1, space_size, params))
        tried_conf.append(params)

        recommender.merge(**params)

        # find out the user in the test set
        to_test = np.diff(test.indptr) > 0
        test_users = np.arange(nusers)[to_test]
        # perform the recommendations
        recommended_items = recommender.recommend(test_users, at)

        # evaluate the ranking quality
        results[i] = evaluate_metric(test, test_users, recommended_items, metric, at)
        logger.info('Result: {:.4f}'.format(results[i]))
    # return the best configuration
    best = results.argsort()[-1]
    return tried_conf[best], results[best]


def grid_search_holdout_rm(RecommenderClass, dataset, nusers, nitems, rec, rec_idx, recommenders, param_space,
                           user_features=None, item_features=None, metric=precision, at=None, perc=0.8,
                           user_key='user_id', item_key='item_id', rnd_seed=1234):
    """
    Find the best hyper-parameters of a recommender algorithm with Grid Search
    """

    tried_conf = []
    results = np.zeros(np.prod([len(v) for v in param_space.values()]), dtype=np.float32)
    space_size = len(results)
    logger.info('Size of the parameter space: {} ({} holdout trials)'.format(space_size, space_size))
    param_grid = ParameterGrid(param_space)
    # compute the holdout split
    train_df, test_df = holdout(dataset,
                                rec,
                                user_key=user_key,
                                item_key=item_key,
                                perc=perc,
                                seed=rnd_seed)

    train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
    test = df_to_csr(test_df, nrows=nusers, ncols=nitems)

    # train the recommender
    recommender = RecommenderClass(recommenders)
    if user_features is not None and item_features is not None:
        recommender.fit(train, rec_idx, user_features, item_features)
    elif user_features is not None:
        recommender.fit(train, rec_idx, user_features)
    elif item_features is not None:
        recommender.fit(train, rec_idx, item_features)
    else:
        recommender.fit(train, rec_idx)

    # find out the user in the test set
    to_test = np.diff(test.indptr) > 0
    test_users = np.arange(nusers)[to_test]
    recommender.inner_recommend(test_users, at)

    for i, params in enumerate(param_grid):
        logger.info('Iteration {}/{}: {}'.format(i+1, space_size, params))
        tried_conf.append(params)

        # perform the recommendations
        recommended_items = recommender.recommend(test_users, at, **params)

        # evaluate the ranking quality
        results[i] = evaluate_metric(test, test_users, recommended_items, metric, at)
        logger.info('Result: {:.4f}'.format(results[i]))
    # return the best configuration
    best = results.argsort()[-1]
    return tried_conf[best], results[best]


def grid_search_holdout_sm(RecommenderClass, dataset, nusers, nitems, rec, rec_idx, recommenders, param_space,
                           user_features=None, item_features=None, metric=precision, at=None, perc=0.8,
                           user_key='user_id', item_key='item_id', rnd_seed=1234):
    """
    Find the best hyper-parameters of a recommender algorithm with Grid Search
    """

    tried_conf = []
    results = np.zeros(np.prod([len(v) for v in param_space.values()]), dtype=np.float32)
    space_size = len(results)
    logger.info('Size of the parameter space: {} ({} holdout trials)'.format(space_size, space_size))
    param_grid = ParameterGrid(param_space)
    # compute the holdout split
    train_df, test_df = holdout(dataset,
                                rec,
                                user_key=user_key,
                                item_key=item_key,
                                perc=perc,
                                seed=rnd_seed)

    train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
    test = df_to_csr(test_df, nrows=nusers, ncols=nitems)

    # train the recommender
    recommender = RecommenderClass(recommenders)
    if user_features is not None and item_features is not None:
        recommender.fit(train, rec_idx, user_features, item_features)
    elif user_features is not None:
        recommender.fit(train, rec_idx, user_features)
    elif item_features is not None:
        recommender.fit(train, rec_idx, item_features)
    else:
        recommender.fit(train, rec_idx)

    # find out the user in the test set
    to_test = np.diff(test.indptr) > 0
    test_users = np.arange(nusers)[to_test]
    recommender.inner_scores(test_users)

    for i, params in enumerate(param_grid):
        logger.info('Iteration {}/{}: {}'.format(i+1, space_size, params))
        tried_conf.append(params)

        # perform the recommendations
        recommended_items = recommender.recommend(test_users, at, **params)

        # evaluate the ranking quality
        results[i] = evaluate_metric(test, test_users, recommended_items, metric, at)
        logger.info('Result: {:.4f}'.format(results[i]))
    # return the best configuration
    best = results.argsort()[-1]
    return tried_conf[best], results[best]


def grid_search_cv(RecommenderClass, dataset, nusers, nitems, rec, rec_idx, param_space, cv_folds, user_features=None,
                   item_features=None, metric=precision, at=None, perc=0.8, user_key='user_id', item_key='item_id',
                   rnd_seed=1234):
    """
    Find the best hyper-parameters of a recommender algorithm with Grid Search
    """

    tried_conf = []
    results = np.zeros(np.prod([len(v) for v in param_space.values()]), dtype=np.float32)
    space_size = len(results)
    logger.info('Size of the parameter space: {} ({} cv trials)'.format(space_size, space_size * cv_folds))
    param_grid = ParameterGrid(param_space)
    # compute the cv splits
    cv_split = []
    for train_df, test_df in k_fold_cv(dataset,
                                       rec,
                                       user_key=user_key,
                                       item_key=item_key,
                                       k=cv_folds,
                                       seed=rnd_seed):
        train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
        test = df_to_csr(test_df, nrows=nusers, ncols=nitems)
        cv_split.append((train, test))

    for i, params in enumerate(param_grid):
        logger.info('Iteration {}/{}: {}'.format(i+1, space_size, params))
        tried_conf.append(params)
        cv_result = 0.0
        for f, (train, test) in enumerate(cv_split):
            # train the recommender
            recommender = RecommenderClass(**params)
            if user_features is not None and item_features is not None:
                recommender.fit(train, rec_idx, user_features, item_features)
            elif user_features is not None:
                recommender.fit(train, rec_idx, user_features)
            elif item_features is not None:
                recommender.fit(train, rec_idx, item_features)
            else:
                recommender.fit(train, rec_idx)

            # find out the user in the test set
            to_test = np.diff(test.indptr) > 0
            test_users = np.arange(nusers)[to_test]
            # perform the recommendations
            recommended_items = recommender.recommend(test_users, at)

            # evaluate the ranking quality
            metric_ = evaluate_metric(test, test_users, recommended_items, metric, at)
            cv_result += metric_
        # average value of the metric in cross-validation
        results[i] = cv_result / cv_folds
        logger.info('Result: {:.4f}'.format(results[i]))
    # return the best configuration
    best = results.argsort()[-1]
    return tried_conf[best], results[best]
