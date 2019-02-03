import time
import uuid
import json
from flask import Blueprint, request, jsonify
from itertools import chain
import redis
import numpy as np
import igraph as ig
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords


redis_store = redis.StrictRedis(host="localhost", port=6379, db=0)


app = Blueprint('tagger', __name__)


@app.route('/summarize_clusters', methods=['GET', 'POST'])
def summarize_clusters():
    """
    list of list of lists of strings
    [
      [
        [cluster1_doc1_token1, cluster1_doc1_token2],
        [cluster1_doc2_token1, cluster1_doc2_token2],
      ],
        [cluster2_doc1_token1, cluster2_doc1_token2],
        [cluster2_doc2_token1, cluster2_doc2_token2],
      ]
    ]
    """
    response = {"success": False}
    if request.method == 'POST':
        r = request.get_json()
        top_n = r.get('top_n')
        lang = r.get('lang')
        method = r.get('method')
        clustered_docs = json.loads(r.get('clustered_docs'))
        if lang == 'en':
            stops = stopwords.words('english')
        if lang == 'de':
            stops = stopwords.words('german')
        else:
            stops = []

        summaries = []
        for cluster in clustered_docs:
            if type(cluster[0]) != str:
                cluster = [c[0] for c in cluster]
            try:
                # get nc embeddings
                doc = list(chain.from_iterable(cluster))
                if type(doc[0]) == list:
                    doc = list(chain.from_iterable(doc))
                doc = list(set(doc))
                k = str(uuid.uuid4())
                d = {"id": k, "doc": doc}
                redis_store.rpush("embed_noun_chunks", json.dumps(d))
                while True:
                    result = redis_store.get(k)
                    if result is not None:
                        result = json.loads(result.decode('utf-8'))
                        embeddings = result.get('embeddings')
                        redis_store.delete(k)
                        break
                    time.sleep(0.5)
                textrank_scores = get_textrank(doc, embeddings)
            except Exception as e:
                textrank_scores = [[1, token] for token in doc]
            df1 = pd.DataFrame(textrank_scores, columns=['textrank', 'token'])
            try:
                tfidf_scores = get_tfidfrank(cluster, stops)
            except Exception as e:
                tfidf_scores = [[1, token] for token in doc]
            df2 = pd.DataFrame(tfidf_scores, columns=['tfidf', 'token'])
            df = pd.merge(df1, df2, on='token')

            # implemented weighted and 2+1 methods here
            summary = get_summary(df, method, weights=(0.5, 0.5), top_n=top_n)
            summaries.append(summary)
        response["summaries"] = summaries
        response["success"] = True
    return jsonify(response)


def get_summary(df, method, weights=(0.5, 0.5), top_n=3):
    """
    scores are rescaled to [0, 1]

    if method is weighted, weights should be a tuple of floats that sum up to 1 (0.25, 0.75)
    the rank of summary candidates is then calculated by weighting the
    tfidf scores with the first value and textrank scores with the second
    the final summary is then the top_n of weighted scores

    if method it n+n, weights should be a tuple of integers (1, 2)
    summary is then the top_n from tfidf scores (first tuple) plus
    the top_n from textrank scores (second tuple)
    """
    df["textrank"] = df.textrank - df.textrank.min()
    df["textrank"] = df.textrank / df.textrank.max()
    df["tfidf"] = df.tfidf - df.tfidf.min()
    df["tfidf"] = df.tfidf / df.tfidf.max()
    df["weighted"] = df.apply(lambda x: x.tfidf * weights[0] +
                                        x.textrank * weights[1], axis=1)
    if method == 'weighted':
        summary = []
        for candidate in df.sort_values('weighted', ascending=False)['token']:
            if candidate.lower() not in [s.lower() for s in summary]:
                if len(candidate) < 35:
                    summary.append(candidate.replace(" - ", "-"))
        summary = ", ".join(summary[:top_n])
        return summary
    if method == 'n+n':
        tfidf_summary = []
        for candidate in df.sort_values('tfidf', ascending=False)['token']:
            if candidate.lower() not in [s.lower() for s in summary]:
                if len(candidate) < 35:
                    tfidf_summary.append(candidate.replace(" - ", "-"))
        textrank_summary = []
        for candidate in df.sort_values('textrank', ascending=False)['token']:
            if candidate.lower() not in [s.lower() for s in summary]:
                if len(candidate) < 35:
                    textrank_summary.append(candidate.replace(" - ", "-"))
        summary = ", ".join(tfidf_summary[:weights[0]] +
                            textrank_summary[:weights[1]])
        return summary


def get_textrank(tokens, embeddings):
    res = [(x, np.array(y))
           for x, y in zip(tokens, embeddings)
           if y is not None]
    # average over noun chunk word vectors for each noun_chunk
    tokens = [r[0] for r in res]
    embeddings = [r[1] for r in res]
    # summarise with textrank
    sim_mat = cosine_similarity(embeddings)
    np.fill_diagonal(sim_mat, 0)
    sources, targets = sim_mat.nonzero()
    conn_indices = np.where(sim_mat)
    weights = sim_mat[conn_indices]
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    G = ig.Graph(edges=list(edgelist), directed=True)
    G.es['weight'] = weights
    scores = G.pagerank(weights="weight")
    ranking = sorted(((scores[i], nc)
                      for i, nc in enumerate(tokens)), reverse=True)
    return ranking


def get_tfidfrank(docs, stops):
    """docs need to be converted from list of lists back to list of string"""
    docs = [" ".join([t.replace(" ", "_") for t in d])
            for d in docs]
    cv = CountVectorizer(max_df=0.85, lowercase=False, stop_words=stops)
    word_count_vector = cv.fit_transform(docs)
    token_names = cv.get_feature_names()
    token_names = [t.replace("_", " ") for t in token_names]
    tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_matrix = tfidf.fit_transform(word_count_vector).tocoo()
    tuples = zip(tfidf_matrix.col, tfidf_matrix.data)
    scores = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    ranking = [(score, token_names[ind]) for ind, score in scores]
    return ranking
