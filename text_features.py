from typing import Iterable
import numpy as np
import pandas as pd

import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


def bag_of_words(texts, cv=None, n_words=1):
    res = [None] * len(texts)
    for i in range(len(texts)):
        words = texts[i].split(' ')
        loc = []
        for j in range(len(words) - n_words + 1):
            loc.append(' '.join(words[j: j + n_words]))
        res[i] = loc

    if cv is None:
        cv = CountVectorizer()
        res = cv.fit_transform(texts)
    else:
        res = cv.transform(texts)
    return res.toarray(), cv

def tf_idf(texts, tfid=None):
    if tfid is None:
        tfid = TfidfVectorizer()
        res = tfid.fit_transform(texts)
    else:
        res = tfid.transform(texts)
    return res.toarray(), tfid

def spacy_approach(texts, nlp=None):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    base_shape = nlp("nlp").vector.shape
    text2vector = lambda x: nlp(x).vector if x else np.zeros(base_shape)
    return np.row_stack([text2vector(text) for text in texts]), nlp

def word2vec_approach(texts):
    texts = [ x.split(' ') for x in texts ]

    model = Word2Vec(texts, 
                        min_count=1,   # Ignore words that appear less than this
                        size=50,      # Dimensionality of word embeddings
                        workers=2,     # Number of processors (parallelisation)
                        window=5,      # Context window for words during training
                        iter=30)       # Number of epochs training over corpus

    max_len = 0
    for sentence in texts:
        if len(sentence) > max_len:
            max_len = len(sentence)

    text_embeds = []
    for sentence in texts:
        sentence_emb = []
        sent_len = len(sentence)
        for word in sentence:
            sentence_emb.append(model.wv[word])
        
        base = (max_len-sent_len) * 50 * [0.0]
        if sent_len < max_len:
            text_embeds.append(base + list(np.concatenate(sentence_emb)))
        else:
            text_embeds.append(np.concatenate(sentence_emb))

    return np.row_stack(text_embeds), model