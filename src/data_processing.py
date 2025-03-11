import numpy as np
import re
from collections import Counter

# Vocabulary dictionary to store token-to-index and index-to-token mappings

vocab = {}

def initialize_vocab():
    """Initializes the vocabulary with a special <UNK> token."""
    unkToken = '<UNK>'
    vocab['t_2_i'] = {}
    vocab['i_2_t'] = {}
    vocab['unkToken'] = unkToken
    idx = add_token(unkToken)
    vocab['unkTokenIdx'] = idx

def add_token(token):
    """Adds a single token to the vocabulary."""
    if token in vocab['t_2_i']:
        idx = vocab['t_2_i'][token]
    else:
        idx = len(vocab['t_2_i'])
        vocab['t_2_i'][token] = idx
        vocab['i_2_t'][idx] = token
    return idx

def add_many_tokens(tokens):
    """Adds multiple tokens to the vocabulary."""
    idxes = [add_token(token) for token in tokens]
    return idxes

def look_up_token(token):
    """Looks up the index of a token in the vocabulary."""
    if vocab['unkTokenIdx'] >= 0:
        return vocab['t_2_i'].get(token, vocab['unkTokenIdx'])
    else:
        return vocab['t_2_i'][token]

def look_up_idx(idx):
    """Looks up the token corresponding to an index in the vocabulary."""
    if idx not in vocab['i_2_t']:
        raise KeyError(f'This index {idx} does not exist in the vocabulary.')
    return vocab['i_2_t'][idx]

def vocab_from_corpus(corpus, cutoff=25):
    """Builds a vocabulary from a text corpus with a frequency cutoff."""
    initialize_vocab()
    wordCounts = Counter()
    for doc in corpus:
        for word in re.split('\W+', doc):
            if word:
                wordCounts[word] += 1
    for word, count in wordCounts.items():
        if count >= cutoff:
            add_token(word)

# Feature engineering functions
def one_hot_vector(token, N):
    """Creates a one-hot vector for a given token."""
    one_hot = np.zeros((N, 1))
    one_hot[look_up_token(token)] = 1
    return one_hot

def compute_features(doc, N):
    """Computes feature vectors for a document."""
    fv = np.zeros(N)
    num_tokens = 0
    for token in doc:
        fv[look_up_token(token)] += 1
        num_tokens += 1
    return fv / num_tokens

def corpus_to_feature_matrix(corpus, N):
    """Converts a corpus into a feature matrix."""
    fM = np.zeros((N, len(corpus)))
    for i, doc in enumerate(corpus):
        fM[:, i] = compute_features(doc, N)
    return fM.T
