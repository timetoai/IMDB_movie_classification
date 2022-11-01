from typing import Iterable
import numpy as np
import pandas as pd

import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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

def spacy_appoach(texts, nlp=None):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    text2vector = lambda x: nlp(x).vector
    return np.row_stack([text2vector(text) for text in texts]), nlp
