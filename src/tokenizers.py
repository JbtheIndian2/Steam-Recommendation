"""
tokenizers.py
Picklable tokenizer/preprocessor functions for the TfidfVectorizer.

These live in their own module (rather than inside train.py) so that when
the fitted vectorizer is joblib-pickled, it references a stable import path
(`tokenizers.soup_tokenizer`) instead of `__main__.soup_tokenizer` — which
would break the moment a *different* entry-point script tries to unpickle
it. This is a common gotcha when a script is both "the thing you run" and
"the thing that defines the function you pickled".
"""


def soup_tokenizer(s):
    """Split the pre-joined '|'-delimited soup into whole tokens, e.g.
    'genre__Action|tag__FPS' -> ['genre__Action', 'tag__FPS']."""
    return s.split('|') if s else []


def identity_preprocessor(s):
    return s
