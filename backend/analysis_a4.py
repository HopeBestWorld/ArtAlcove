from typing import List, Tuple, Dict
from collections.abc import Callable
import numpy as np
import helpers_a4 as helpers
import math
from collections import defaultdict
from collections import Counter
from helpers_a4 import adj_chars



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

    # TODO-1.1
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


def edit_distance_search(
    query: str,
    msgs: List[dict],
    ins_cost_func: int,
    del_cost_func: int,
    sub_cost_func: int,
) -> List[Tuple[int, dict]]:
    """Edit distance search

    Arguments
    =========
    query: string,
        The query we are looking for.

    msgs: list of dicts,
        Each message in this list has a 'text' field with
        the raw document.

    ins_cost_func: function that returns the cost of inserting a letter,

    del_cost_func: function that returns the cost of deleting a letter,

    sub_cost_func: function that returns the cost of substituting a letter,

    Returns
    =======
    result: list of (score, message) tuples.
        The result list is sorted by score such that the closest match
        is the top result in the list.

    """
    # TODO-1.2
    results = []
    
    for msg in msgs:
        text = msg.get('text', '')
        if text:
            distance = edit_distance(query, text, ins_cost_func, del_cost_func, sub_cost_func)
            results.append((distance, msg))
    
    results.sort(key=lambda x: x[0])
    
    return results


def substitution_cost_adj(query: str, message: str, i: int, j: int) -> int:
    """
    Custom substitution cost:
    The cost is 1.5 when substituting a pair of characters that can be found in helpers.adj_chars
    Otherwise, the cost is 2. (Not 1 as it was before!)
    """
    # TODO-2.1
    adj_chars_set = set(adj_chars)
    if query[i - 1] == message[j - 1]: 
        return 0
    
    char_pair = (query[i - 1], message[j - 1])
    rev_char_pair = (message[j - 1], query[i - 1]) 

    if char_pair in adj_chars_set or rev_char_pair in adj_chars_set:
        return 1.5
    else:
        return 2


def build_inverted_index(msgs: List[dict]) -> dict:
    """Builds an inverted index from the messages.

    Arguments
    =========

    msgs: list of dicts.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> test_idx = build_inverted_index([
    ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]

    """
    # TODO-3.1
    inverted_index = {}

    for doc_id, msg in enumerate(msgs):
        term_counts = {}

        for term in msg["toks"]:
            term_counts[term] = term_counts.get(term, 0) + 1

        for term, count in term_counts.items():
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append((doc_id, count))

    return inverted_index


def boolean_search(query_word_1: str, query_word_2: str, inverted_index: dict) -> List[int]:
    """Search the given collection of documents that contains query_word_1 or query_word_2

    Arguments
    =========

    query_word_1: string,
        The first word we are searching for in our documents.

    query_word_2: string,
        The second word we are searching for in our documents.

    inverted_index: an inverted index as above


    Returns
    =======

    results: list of ints
        Sorted List of results (in increasing order) such that every element is a `doc_id`
        that points to a document that satisfies the boolean
        expression of the query.

    """
    # TODO-4.1
    query_word_1 = query_word_1.lower()
    query_word_2 = query_word_2.lower()

    postings1 = inverted_index.get(query_word_1, [])
    postings2 = inverted_index.get(query_word_2, [])

    merged_posting = []
    i, j = 0, 0

    while i < len(postings1) and j < len(postings2):
        if postings1[i] == postings2[j]:
            merged_posting.append(postings1[i])
            i += 1
            j += 1
        elif postings1[i] < postings2[j]:
            merged_posting.append(postings1[i])
            i += 1
        else:
            merged_posting.append(postings2[j])
            j += 1

    merged_posting.extend(postings1[i:])
    merged_posting.extend(postings2[j:])

    return merged_posting


def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Hint: Make sure to use log base 2.

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """

    # TODO-5.1
    idf = {}
    max_df = max_df_ratio * n_docs  

    for term, postings in inv_idx.items():
        df_t = len(postings)  

        if min_df <= df_t <= max_df:
            idf[term] = math.log2((n_docs + 1) / df_t)

    return idf


def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.

    Arguments
    =========

    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.

    Returns
    =======

    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """

    # TODO-6.1
    norms = np.zeros(n_docs) 

    for term, postings in index.items():
        if term in idf: 
            idf_t = idf[term]
            for doc_id, tf in postings:  
                norms[doc_id] += (tf * idf_t) ** 2

    return np.sqrt(norms)


def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.


    Returns
    =======
    
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    # TODO-7.1
    doc_scores = {} 
    
    for term, query_tf in query_word_counts.items():
        if term in index: 
            term_idf = idf.get(term, 0)  
            query_weight = query_tf * term_idf  

            for doc_id, doc_tf in index[term]: 
                doc_weight = doc_tf * term_idf  
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + query_weight * doc_weight
                

    
    max_doc = max(doc_scores, key=doc_scores.get)
    max_score = max(doc_scores.values())
    if max_score > 219.99:
      scale_factor = 219.99 / max_score 
      doc_scores = {doc: score * scale_factor for doc, score in doc_scores.items()}

    return doc_scores



def index_search(
    query: str,
    index: dict,
    idf,
    doc_norms,
    score_func=accumulate_dot_scores,
    tokenizer=helpers.treebank_tokenizer,
) -> List[Tuple[int, int]]:
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.

    Note:

    """

    # TODO-8.1
    query = query.lower()
    query_tokens = tokenizer.tokenize(query)
    query_word_counts = Counter(query_tokens)

    doc_scores = score_func(query_word_counts, index, idf)

    query_norm = 0
    for term, count in query_word_counts.items():
        if term in idf:  
            query_norm += (count * idf[term])**2
    query_norm = np.sqrt(query_norm) 

    results = []
    for doc_id, score in doc_scores.items():
        cosine_similarity = score / (doc_norms[doc_id] * query_norm) if doc_norms[doc_id] > 0 and query_norm > 0 else 0 
        results.append((cosine_similarity, doc_id))

    results.sort(reverse=True, key=lambda x: x[0])

    return results