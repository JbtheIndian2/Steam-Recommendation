"""
preprocess.py
Cleans the raw Steam Store Games dataset into a modeling-ready table.

Input:  data/steam.csv        (raw Kaggle "Steam Store Games" export)
Output: data/games_clean.csv  (filtered, feature-engineered table)

Filtering rationale:
  - english == 1        -> keep titles with English metadata (tag/genre text is English)
  - review_count >= 100 -> drop asset-flips / shovelware with too little signal to trust
"""
import pandas as pd
import numpy as np

MIN_REVIEWS = 100

def split_field(s):
    if not isinstance(s, str) or not s:
        return []
    return [t.strip() for t in s.split(';') if t.strip()]

# categories that carry taste/preference signal (vs. store-page boilerplate
# like "Steam Achievements" or "Steam Cloud", which we deliberately drop)
KEEP_CATEGORIES = {
    'Co-op', 'Online Co-op', 'Local Co-op', 'Multi-player', 'Single-player',
    'Online Multi-Player', 'Local Multi-Player', 'MMO', 'VR Support',
    'Split Screen', 'Cross-Platform Multiplayer'
}


def load_and_clean(path='data/steam.csv'):
    df = pd.read_csv(path)

    df = df[df['english'] == 1].copy()
    df['review_count'] = df['positive_ratings'] + df['negative_ratings']
    df = df[df['review_count'] >= MIN_REVIEWS].copy()

    df['rating'] = (df['positive_ratings'] / df['review_count'] * 100).round(1)
    df['release_year'] = df['release_date'].str[:4]

    df['genre_list'] = df['genres'].apply(split_field)
    df['tag_list'] = df['steamspy_tags'].apply(split_field)
    df['cat_list'] = df['categories'].apply(
        lambda s: [c for c in split_field(s) if c in KEEP_CATEGORIES]
    )

    def owners_midpoint(s):
        try:
            lo, hi = s.split('-')
            return (int(lo) + int(hi)) // 2
        except Exception:
            return 0
    df['owners_mid'] = df['owners'].apply(owners_midpoint)

    # the "content soup": every genre/tag/category becomes ONE token
    # (not word-split) so multi-word tags like "Turn-Based Strategy" or
    # "Bullet Hell" survive intact. Prefixing with a namespace (genre__ /
    # tag__ / cat__) keeps identical-looking words in different fields from
    # colliding, e.g. genre "Action" vs tag "Action" are distinct signals.
    def make_soup(row):
        tokens = (
            [f'genre__{g}' for g in row['genre_list']] +
            [f'tag__{t}' for t in row['tag_list']] +
            [f'cat__{c}' for c in row['cat_list']]
        )
        return '|'.join(tokens)

    df['soup'] = df.apply(make_soup, axis=1)
    df = df[df['soup'].str.len() > 0].reset_index(drop=True)

    out = df[[
        'appid', 'name', 'developer', 'publisher', 'release_year', 'price',
        'rating', 'review_count', 'owners_mid', 'genre_list', 'tag_list',
        'cat_list', 'soup'
    ]].rename(columns={'appid': 'id', 'name': 'n', 'owners_mid': 'owners'})

    return out


if __name__ == '__main__':
    games = load_and_clean()
    games.to_csv('data/games_clean.csv', index=False)
    print(f'{len(games)} games kept (of raw dataset) after filtering to '
          f'reviews >= {MIN_REVIEWS} and english == 1')
    print(games[['n', 'genre_list', 'tag_list', 'rating']].head())
