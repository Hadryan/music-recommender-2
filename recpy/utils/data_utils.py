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
import pandas as pd
from sklearn.preprocessing import normalize

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(name)s: %(levelname)s: %(message)s')


def read_profile(path,
                 header=None,
                 columns=None,
                 key='id',
                 sep=','):
    data = pd.read_csv(path, header=header, names=columns, sep=sep)
    logger.info('Columns: {}'.format(data.columns.values))
    # build user (or item) maps and reverse maps
    # this is used to map ids to indexes starting from 0 to nusers (or nitems)
    keys = data[key].unique()
    key_to_idx = pd.Series(data=np.arange(len(keys)), index=keys)
    # for some reason the following doesn't work in python 2.7
    #idx_to_key = pd.Series(index=key_to_idx.data, data=key_to_idx.index)
    idx_to_key = pd.Series(index=key_to_idx.values, data=key_to_idx.index)
    return data, key_to_idx, idx_to_key


def tfidf(partial):
    
    num_tot_documents = partial.shape[0]

    partial = partial.tocsr()
    # count how many features have a certain document
    features_per_document = np.asarray(partial.sum(axis=1)).ravel()

    def tf(n_features):
        if n_features:
            return 1/n_features
        else:
            return 0

    tf = [tf(n_features) for n_features in features_per_document]
    row_nnz = np.diff(partial.indptr)
    # normalize the values in each row
    partial.data *= np.repeat(tf, row_nnz)
    partial = partial.tocsc()

    # count how many documents have a certain feature
    documents_per_feature = np.asarray(partial.sum(axis=0)).ravel()
    idf = np.log(num_tot_documents/documents_per_feature, dtype=np.float32)
    col_nnz = np.diff(partial.indptr)
    # normalize the values in each col
    partial.data *= np.repeat(idf, col_nnz)

    return partial


def profile_to_ucm(user_profile, user_to_idx):

    n_users = user_profile.shape[0]

    # Create the partial UCM on the 'owner' attribute
    owners = user_profile['owner'].unique()  # cannot be used as index
    n_owners = owners.shape[0]
    owner_to_idx = pd.Series(data=np.arange(n_owners), index=owners)

    rows = user_to_idx[user_profile['playlist_id']]
    columns = owner_to_idx[user_profile['owner']]
    values = np.ones(n_users)
    shape = (n_users, n_owners)
    partial_on_owner = sps.csc_matrix((values, (rows, columns)), shape=shape, dtype=np.float32)

    # Create the partial UCM on the 'title' attribute
    titles = user_profile['title'].apply(lambda s: s[1:-1].split(", "))  # can be used as index
    words = np.unique([word for title in titles for word in title])
    n_words = words.shape[0] - 1  # to avoid adding a column for ''

    rows, columns, values = [], [], []
    for i in range(n_users):
        title = titles[i]
        for word in title:
            try:
                int(word)
                rows.append(user_to_idx[user_profile['playlist_id'][i]])
                columns.append(int(word))
                values.append(1)
            except ValueError:
                pass

    shape = (n_users, n_words)
    partial_on_title = sps.csc_matrix((values, (rows, columns)), shape=shape, dtype=np.float32)
    # Apply TF-IDF
    #tfidf(partial_on_title)
    #partial_on_title = normalize(partial_on_title, norm='l2', copy=False)  # with l1 it gets worse

    # Create UCM (concatenate all attribute matrices)
    ucm = sps.hstack((partial_on_owner,partial_on_title))

    return ucm.tocsr()


def profile_to_icm(item_profile, item_to_idx):

    n_users = item_profile.shape[0]

    # Create the partial ICM on the 'artist_id' attribute
    artists = item_profile['artist_id'].unique()  # cannot be used as index
    n_artists = artists.shape[0]
    artists_to_idx = pd.Series(data=np.arange(n_artists), index=artists)

    rows = item_to_idx[item_profile['track_id']]
    columns = artists_to_idx[item_profile['artist_id']]
    values = np.ones(n_users)
    shape = (n_users, n_artists)
    partial_on_artists = sps.csc_matrix((values, (rows, columns)), shape=shape, dtype=np.float32)

    # Create the partial ICM on the 'album' attribute
    albums = item_profile['album'].apply(lambda s: s[1:-1])  # cannot be used as index
    albums_unique = np.unique(albums)[1:-1]  # to avoid adding a column for '' and 'None' (np.unique also sorts the elements)
    n_albums = albums_unique.shape[0]

    albums_to_idx = pd.Series(data=np.arange(n_albums), index=albums_unique)

    rows, columns, values = [], [], []
    for i in range(n_users):
        try:
            int(albums[i])
            rows.append(item_to_idx[item_profile['track_id'][i]])
            columns.append(albums_to_idx[albums[i]])
            values.append(1)
        except ValueError:
            pass

    shape = (n_users, n_albums)
    partial_on_albums = sps.csc_matrix((values, (rows, columns)), shape=shape, dtype=np.float32)

    # Create the partial ICM on the 'tags' attribute
    tagss = item_profile['tags'].apply(lambda s: s[1:-1].split(", "))  # cannot be used as index
    tags = np.unique([tag for tags in tagss for tag in tags])[1:]  # to avoid adding a column for '' (np.unique also sorts the elements)
    n_tags = tags.shape[0]

    tags_to_idx = pd.Series(data=np.arange(n_tags), index=tags)

    rows, columns, values = [], [], []
    for i in range(n_users):
        tags = tagss[i]
        for tag in tags:
            try:
                int(tag)
                rows.append(item_to_idx[item_profile['track_id'][i]])
                columns.append(tags_to_idx[tag])
                values.append(1)
            except ValueError:
                pass

    shape = (n_users, n_tags)
    partial_on_tags = sps.csc_matrix((values, (rows, columns)), shape=shape, dtype=np.float32)
    # Apply TF-IDF
    #tfidf(partial_on_tags)
    #partial_on_tags = normalize(partial_on_tags, norm='l2', copy=False)  # with l1 it gets worse

    # Create ICM (concatenate all attribute matrices)
    icm = sps.hstack((partial_on_artists, partial_on_albums, partial_on_tags))

    return icm.tocsr()


def read_dataset(path, header=None,
                 columns=None,
                 user_key='user_id',
                 item_key='item_id',
                 sep=',',
                 user_to_idx=None,
                 item_to_idx=None):
    data = pd.read_csv(path, header=header, names=columns, sep=sep)
    logger.info('Columns: {}'.format(data.columns.values))
    if not ('item_idx' in data.columns and 'user_idx' in data.columns):
        # build user and item maps (and reverse maps)
        # this is used to map ids to indexes starting from 0 to nitems (or nusers)
        items = data[item_key].unique()
        users = data[user_key].unique()
        if item_to_idx is None:
            item_to_idx = pd.Series(data=np.arange(len(items)), index=items)
        if user_to_idx is None:
            user_to_idx = pd.Series(data=np.arange(len(users)), index=users)
        # for some reason the following doesn't work in python 2.7
        #idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
        idx_to_item = pd.Series(index=item_to_idx.values, data=item_to_idx.index)
        # for some reason the following doesn't work in python 2.7
        #idx_to_user = pd.Series(index=user_to_idx.data, data=user_to_idx.index)
        idx_to_user = pd.Series(index=user_to_idx.values, data=user_to_idx.index)
        # map ids to indices
        data['item_idx'] = item_to_idx[data[item_key].values].values
        data['user_idx'] = user_to_idx[data[user_key].values].values
        return data, idx_to_user, idx_to_item
    else:
        return data


def df_to_csr(df, nrows, ncols, user_key='user_idx', item_key='item_idx'):  # dataset_to_urm
    """
    Convert a pandas DataFrame to a scipy.sparse.csr_matrix
    """

    rows = df[user_key].values
    columns = df[item_key].values
    ratings = np.ones(df.shape[0])
    shape = (nrows, ncols)
    # using the 4th constructor of csr_matrix
    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    csr = sps.csr_matrix((ratings, (rows, columns)), shape=shape)

    return csr
