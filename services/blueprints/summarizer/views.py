import time
import uuid
import json
from flask import Blueprint, request, jsonify
from itertools import chain
import redis
import numpy as np
import igraph as ig
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
                doc = list(chain.from_iterable(cluster))
                doc = list(chain.from_iterable(doc))
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
                ranked_tokens = get_textrank(doc, embeddings)
            except Exception as e:
                try:
                    ranked_tokens = get_tfidfrank(cluster, stops)
                except Exception as e:
                    ranked_tokens = [[0, ""]]
            summary = []
            for rnc in ranked_tokens:
                candidate = rnc[1]
                if candidate.lower() not in [s.lower() for s in summary]:
                    if len(candidate) < 35:
                        summary.append(candidate.replace(" - ", "-"))
            summaries.append(", ".join(summary[:top_n]))
        response["summaries"] = summaries
        response["success"] = True
    return jsonify(response)


@app.route('/summarize', methods=['GET', 'POST'])
def summarize_doc():
    # doc
    """
    if method is textrank, doc should be a list of noun chunks
    """
    response = {"success": False}
    if request.method == 'POST':
        r = request.get_json()
        method = r.get('method')
        top_n = r.get('top_n')
        doc = json.loads(r.get('doc'))

        if method == 'textrank':
            # get nc embeddings
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
            try:
                ranked_tokens = get_textrank(doc, embeddings)
                summary = []
                for rnc in ranked_tokens:
                    candidate = rnc[1]
                    if candidate.lower() not in [s.lower() for s in summary]:
                        if len(candidate) < 35:
                            summary.append(candidate.replace(" - ", "-"))
                response["summary"] = ", ".join(summary[:top_n])
                response["success"] = True
            except Exception as e:
                response["msg"] = str(e)
                response["summary"] = None
        elif method == 'tfidf':
            response["msg"] = "Method not implemented, choose one of ['textrank']"
        else:
            response["msg"] = "Method not implemented, choose one of ['textrank']"

    return jsonify(response)


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
