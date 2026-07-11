"""
eda.py
Quick exploratory analysis of the cleaned dataset. Run after preprocess.py.
Writes a few charts to reports/ and prints summary stats to stdout —
useful both as a sanity check on the cleaning step and as portfolio
evidence of the analysis behind the modeling choices.
"""
import ast
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'reports'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv('data/games_clean.csv')
    df['genre_list'] = df['genre_list'].apply(ast.literal_eval)
    df['tag_list'] = df['tag_list'].apply(ast.literal_eval)

    print(f'Games: {len(df)}')
    print(f'Median price: ${df["price"].median():.2f}')
    print(f'Median rating: {df["rating"].median()}%')
    print(f'Free-to-play: {(df["price"] == 0).mean():.1%}')

    # top genres
    genre_counts = pd.Series([g for row in df['genre_list'] for g in row]).value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    genre_counts.sort_values().plot.barh(ax=ax, color='#e8a33d')
    ax.set_title('Most common genres (top 15)')
    ax.set_xlabel('# games')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/top_genres.png', dpi=140)

    # top tags
    tag_counts = pd.Series([t for row in df['tag_list'] for t in row]).value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    tag_counts.sort_values().plot.barh(ax=ax, color='#4fd1c5')
    ax.set_title('Most common community tags (top 15)')
    ax.set_xlabel('# games')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/top_tags.png', dpi=140)

    # rating distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df['rating'].plot.hist(bins=40, ax=ax, color='#d9707c')
    ax.set_title('Distribution of positive-review rating')
    ax.set_xlabel('% positive')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/rating_distribution.png', dpi=140)

    print(f'\nCharts written to {OUT_DIR}/')


if __name__ == '__main__':
    main()
