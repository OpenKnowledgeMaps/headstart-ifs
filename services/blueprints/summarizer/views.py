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
    a list of clustered documents, where each document is a list of tokens
    or noun chunks
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
            try:
                # get nc embeddings
                doc = list(set(chain.from_iterable(cluster)))
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
                print(e)
                textrank_scores = [[1, token] for token in doc]
            df1 = pd.DataFrame(textrank_scores, columns=['textrank', 'token'])
            try:
                tfidf_scores = get_tfidfrank(cluster, stops)
            except Exception as e:
                print(e)
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
    df["textrank"] = df.textrank - df.textrank.min()
    df["textrank"] = df.textrank / df.textrank.max()
    df["tfidf"] = df.tfidf - df.tfidf.min()
    df["tfidf"] = df.tfidf / df.tfidf.max()
    df["weighted"] = df.apply(lambda x: x.textrank * weights[0] +
                                        x.tfidf * weights[1], axis=1)
    if method == 'weighted':
        summary = []
        for candidate in df.sort_values('weighted', ascending=False)['token']:
            if candidate.lower() not in [s.lower() for s in summary]:
                if len(candidate) < 35:
                    summary.append(candidate.replace(" - ", "-"))
        summary = ", ".join(summary[:top_n])
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
