import numpy as np
import re
import math
from collections import Counter

def tokenize(text):
    """tokenizes input text into words & converts to lowercase."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def compute_tf(doc_tokens):
    """computes term frequency for a list of tokens."""
    term_counts = Counter(doc_tokens)
    total_terms = len(doc_tokens)
    return {term: count / total_terms for term, count in term_counts.items()}

def compute_idf(docs):
    """computes IDF for a collection of documents."""
    num_docs = len(docs)
    term_doc_counts = Counter(term for doc in docs for term in set(doc))
    return {term: math.log(num_docs / (1 + count)) for term, count in term_doc_counts.items()}

def compute_tfidf(docs):
    """computes TF-IDF matrix for a list of tokenized documents."""
    idf = compute_idf(docs)
    return [{term: tf * idf[term] for term, tf in compute_tf(doc).items()} for doc in docs]

def query(q, docs):
    """processes a query and retrieves relevant documents using TF-IDF."""
    q_tokens = tokenize(q)
    docs_tokens = [tokenize(doc) for doc in docs]
    tfidf_docs = compute_tfidf(docs_tokens)
    q_tfidf = compute_tf(q_tokens)
    
    scores = []
    for i, doc_tfidf in enumerate(tfidf_docs):
        score = sum(q_tfidf.get(term, 0) * doc_tfidf.get(term, 0) for term in q_tokens)
        scores.append((i, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def compute_tfidf(docs):
    """Computes TF-IDF matrix for a list of tokenized documents."""
    idf = compute_idf(docs)
    return [{term: tf * idf[term] for term, tf in compute_tf(doc).items()} for doc in docs]

def query_json_search(query, merged_df):
    """Searches for products using TF-IDF and returns results in JSON format."""
    
    merged_df["search_text"] = (
        merged_df["descr"].astype(str) + " " + 
        merged_df["review_title"].astype(str) + " " + 
        merged_df["review_desc"].astype(str)
    )
    
    docs_tokens = [tokenize(text) for text in merged_df["search_text"]]
    
    tfidf_docs = compute_tfidf(docs_tokens)
    
    q_tokens = tokenize(query)
    q_tfidf = compute_tf(q_tokens)
    
    scores = []
    for i, doc_tfidf in enumerate(tfidf_docs):
        score = sum(q_tfidf.get(term, 0) * doc_tfidf.get(term, 0) for term in q_tokens)
        if score > 0:  # Only include relevant matches
            scores.append((i, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    matches = merged_df.iloc[[i for i, _ in scores]]

    grouped_matches = matches.groupby(
        ['product', 'siteurl', 'price', 'rating', 'imgurl', 'descr']
    ).agg({
        'review_title': list,
        'review_desc': list
    }).reset_index()

    return grouped_matches.to_json(orient='records')