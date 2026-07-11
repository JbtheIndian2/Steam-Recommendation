"""
train.py
Fits the content-based recommendation model with scikit-learn and
serializes the artifacts needed to serve recommendations later.

Model:
  1. TfidfVectorizer turns each game's tag/genre/category "soup" into a
     sparse weighted vector. Because rare tags (e.g. "Metroidvania") are
     more descriptive than common ones (e.g. "Indie"), TF-IDF naturally
     downweights the common ones via the inverse-document-frequency term:
         idf(t) = ln((1 + N) / (1 + df(t))) + 1
  2. NearestNeighbors(metric='cosine') indexes those vectors so that,
     given a game, we can retrieve its closest neighbors in tag-space in
     sub-millisecond time.

Output artifacts (models/):
  - tfidf_vectorizer.joblib
  - nn_model.joblib
  - tfidf_matrix.npz         (sparse game x vocabulary matrix)
  - games_meta.csv           (display metadata, row-aligned with the matrix)
"""
import pandas as pd
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tokenizers import soup_tokenizer, identity_preprocessor


def train(games_path='data/games_clean.csv', out_dir='models'):
    games = pd.read_csv(games_path).reset_index(drop=True)
    games['soup'] = games['soup'].fillna('')

    # each token in the soup is already a full tag (namespaced, e.g.
    # "tag__Roguelike") separated by '|', so we split on '|' rather than
    # letting TfidfVectorizer tokenize on whitespace/punctuation - that
    # would shred multi-word tags like "Turn-Based Strategy".
    vectorizer = TfidfVectorizer(
        tokenizer=soup_tokenizer,
        preprocessor=identity_preprocessor,
        token_pattern=None,
        lowercase=False,
        min_df=2,          # drop tokens that appear in fewer than 2 games
    )
    tfidf_matrix = vectorizer.fit_transform(games['soup'])

    nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix)

    joblib.dump(vectorizer, f'{out_dir}/tfidf_vectorizer.joblib')
    joblib.dump(nn_model, f'{out_dir}/nn_model.joblib')
    sparse.save_npz(f'{out_dir}/tfidf_matrix.npz', tfidf_matrix)
    games.drop(columns=['soup']).to_csv(f'{out_dir}/games_meta.csv', index=False)

    print(f'Vocabulary size: {len(vectorizer.vocabulary_)}')
    print(f'Matrix shape: {tfidf_matrix.shape}  (games x vocabulary)')
    print(f'Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0]*tfidf_matrix.shape[1]):.4%}')
    print(f'Artifacts written to {out_dir}/')


if __name__ == '__main__':
    train()
