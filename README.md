*This project was realized as part of the competition within the Politecnico di Milano course in Recommender systems*

# music-recommender

Music streaming services allow users to listen to songs and to create playlists of favorite songs.

This recommender system suggests songs a user would likely add to one of her playlists based on:

- other tracks in the same playlist
- other playlists created by the same user
- other playlists created by other users

## Competition

The original unsplitted dataset includes around 1M interactions (tracks belonging to a playlist) for 57k playlists and 100k items (tracks). A subset of about 10k playlists and 32k items has been selected as test playlists and items.

The goal is to recommend a list of 5 relevant items for each playlist. MAP@5 is used for evaluation.

No restrictions on the recommender algorithm choice (e.g., collaborative-filtering, content-based, hybrid, etc.) written in any language.

## Model tuning

Hold-out

```shell
make tune_with_holdout
```

Cross-validation

```shell
make tune_with_cv
```

## Model evaluation

Hold-out

```shell
make holdout
```

Cross-validation

```shell
make cv
```

## Recommendation

To generate the recommendations

```shell
make run
```
