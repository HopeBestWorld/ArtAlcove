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
    
def edit_distance(
    query: str, message: str, ins_cost_func: int, del_cost_func: int, sub_cost_func: int
) -> int:
    """Finds the edit distance between a query and a message using the edit matrix

    Arguments
    =========
    query: query string,

    message: message string,

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns:
        edit cost (int)
    """

    query = query.lower()
    message = message.lower()

    m, n = len(query), len(message)
    dp = np.zeros((m+1, n+1), dtype=float)


    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + del_cost_func(query, i)  
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + ins_cost_func(message, j)

    for i in range(1, m+1):
        for j in range(1, n+1):
            insertion = dp[i][j-1] + ins_cost_func(message, j)
            deletion = dp[i-1][j] + del_cost_func(query, i)
            substitution = dp[i-1][j-1] + sub_cost_func(query, message, i, j)
            
            dp[i][j] = min(insertion, deletion, substitution)

    return dp[m][n]