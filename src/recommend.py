"""
recommend.py
Loads the trained TF-IDF + NearestNeighbors artifacts and serves
recommendations. Used by both the CLI (`python src/recommend.py "Portal 2"`)
and the Flask API (app.py).
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from tokenizers import soup_tokenizer, identity_preprocessor  # noqa: F401  (needed for unpickling)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


class Recommender:
    def __init__(self, models_dir=MODELS_DIR):
        self.vectorizer = joblib.load(f'{models_dir}/tfidf_vectorizer.joblib')
        self.nn_model = joblib.load(f'{models_dir}/nn_model.joblib')
        self.matrix = sparse.load_npz(f'{models_dir}/tfidf_matrix.npz')
        self.games = pd.read_csv(f'{models_dir}/games_meta.csv')
        self.id_to_row = {gid: i for i, gid in enumerate(self.games['id'])}
        self.vocab_inv = {v: k for k, v in self.vectorizer.vocabulary_.items()}

    def search(self, query, limit=8):
        """Simple substring search over game titles, ranked by review_count."""
        q = query.lower().strip()
        if not q:
            return []
        mask = self.games['n'].str.lower().str.contains(q, regex=False)
        results = self.games[mask].sort_values('review_count', ascending=False)
        return results.head(limit).to_dict('records')

    def game_by_id(self, game_id):
        row = self.games[self.games['id'] == game_id]
        return row.iloc[0].to_dict() if len(row) else None

    def shared_tokens(self, row_a, row_b, limit=3):
        """Return the top-weighted tokens two games have in common, used to
        explain *why* a recommendation was made."""
        vec_a = self.matrix[row_a]
        vec_b = self.matrix[row_b]
        overlap = vec_a.multiply(vec_b)
        overlap = np.asarray(overlap.todense()).flatten()
        top_idx = overlap.argsort()[::-1]
        tokens = []
        seen_labels = set()
        for idx in top_idx:
            if overlap[idx] <= 0 or len(tokens) >= limit:
                break
            raw = self.vocab_inv[idx]          # e.g. "tag__Roguelike"
            label = raw.split('__', 1)[1] if '__' in raw else raw
            if label in seen_labels:           # same label can appear as both
                continue                       # a genre and a tag - show once
            seen_labels.add(label)
            tokens.append(label)
        return tokens

    def recommend(self, game_id, k=12, min_rating=0, price_mode='any', diff_dev_only=False):
        if game_id not in self.id_to_row:
            return None
        row = self.id_to_row[game_id]
        query_vec = self.matrix[row]

        # over-fetch candidates so post-hoc filters (rating/price/developer)
        # still leave us with k results after trimming
        n_candidates = min(len(self.games), max(k * 6, 60))
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=n_candidates)

        query_dev = self.games.iloc[row]['developer']
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == row:
                continue
            g = self.games.iloc[idx]
            if g['rating'] < min_rating:
                continue
            if price_mode == 'free' and g['price'] > 0:
                continue
            if price_mode == 'paid' and g['price'] == 0:
                continue
            if diff_dev_only and g['developer'] == query_dev:
                continue
            similarity = 1 - dist  # cosine distance -> cosine similarity
            results.append({
                **g.to_dict(),
                'similarity': round(float(similarity), 4),
                'shared_tokens': self.shared_tokens(row, idx),
            })
            if len(results) >= k:
                break
        return results


if __name__ == '__main__':
    rec = Recommender()
    query = ' '.join(sys.argv[1:]) or 'Portal 2'
    matches = rec.search(query, limit=1)
    if not matches:
        print(f'No game found matching "{query}"')
        sys.exit(1)
    game = matches[0]
    print(f"Selected: {game['n']} ({game['release_year']}) — {game['rating']}% positive\n")
    recs = rec.recommend(game['id'], k=10)
    print(f"{'Rank':<5}{'Match':<8}{'Game':<38}{'Shared tags'}")
    for i, r in enumerate(recs, 1):
        print(f"{i:<5}{r['similarity']*100:>5.1f}%  {r['n'][:36]:<38}{', '.join(r['shared_tokens'])}")
