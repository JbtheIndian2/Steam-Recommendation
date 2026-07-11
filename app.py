"""
app.py
Flask server: exposes the trained recommender as a JSON API and serves the
static frontend. Run with:

    python app.py

then open http://localhost:5000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, jsonify, request, send_from_directory
from recommend import Recommender

app = Flask(__name__, static_folder='static', static_url_path='')
rec = Recommender()


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/meta')
def meta():
    return jsonify({
        'n_games': len(rec.games),
        'vocab_size': len(rec.vectorizer.vocabulary_),
    })


@app.route('/api/search')
def search():
    q = request.args.get('q', '')
    limit = int(request.args.get('limit', 8))
    results = rec.search(q, limit=limit)
    return jsonify(clean_records(results))


@app.route('/api/recommend/<int:game_id>')
def recommend(game_id):
    k = int(request.args.get('k', 12))
    min_rating = float(request.args.get('min_rating', 0))
    price_mode = request.args.get('price', 'any')
    diff_dev_only = request.args.get('diff_dev', 'false') == 'true'

    game = rec.game_by_id(game_id)
    if game is None:
        return jsonify({'error': 'game not found'}), 404

    recs = rec.recommend(
        game_id, k=k, min_rating=min_rating,
        price_mode=price_mode, diff_dev_only=diff_dev_only,
    )
    return jsonify({
        'game': clean_record(game),
        'recommendations': clean_records(recs),
    })


def clean_record(d):
    """Make a dict JSON-safe (numpy/pandas types -> native python) and
    parse the stringified list columns back into real lists."""
    import ast
    import numpy as np
    out = {}
    for k, v in d.items():
        if k in ('genre_list', 'tag_list', 'cat_list') and isinstance(v, str):
            try:
                out[k] = ast.literal_eval(v)
            except Exception:
                out[k] = []
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, float) and str(v) == 'nan':
            out[k] = None
        else:
            out[k] = v
    return out


def clean_records(records):
    return [clean_record(r) for r in records]


if __name__ == '__main__':
    app.run(debug=True, port=5000)
