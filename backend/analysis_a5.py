from typing import List, Tuple, Dict
from collections.abc import Callable
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy import linalg as LA

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    """Returns a TfidfVectorizer object with the above preprocessing properties.
    
    Note: This function may log a deprecation warning. This is normal, and you
    can simply ignore it.
    
    Parameters
    ----------
    max_features : int
        Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer 
        constructer.
    stop_words : str
        Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer. 
    max_df : float
        Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer. 
    min_df : float
        Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer. 
    norm : str
        Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer. 

    Returns
    -------
    TfidfVectorizer
        A TfidfVectorizer object with the given parameters as its preprocessing properties.
    """
    # TODO-5.1
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        max_df=max_df,
        min_df=min_df,
        norm=norm
    )
    return tfidf_vectorizer

def get_sim(vector1, vector2): # Changed parameter names
    """Returns cosine similarity of two vectors."""
    v1 = np.array(vector1) # Now directly accepts array-like input
    v2 = np.array(vector2) # Now directly accepts array-like input
    n = np.dot(v1, v2)
    d = LA.norm(v1) * LA.norm(v2)
    if d == 0:
        return 0.0
    else:
        return n / d