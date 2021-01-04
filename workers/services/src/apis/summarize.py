import time
import uuid
import json
from flask import request, jsonify, make_response
from flask_restx import Namespace, Resource, fields
from itertools import chain
import redis
import numpy as np
import msgpack_numpy as mnp
import igraph as ig
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
import requests


mnp.patch()
with open("redis_config.json") as infile:
    redis_config = json.load(infile)

redis_store = redis.StrictRedis(**redis_config)
embed_url = "http://localhost/ifs/enrich/sent_embed/gusem"

summarization_ns = Namespace("summarize", description="OKMAps summarization operations")


@summarization_ns.route('/clusters')
class SummarizeClusters(Resource):
    def post(self):
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
        result = {"success": False}
        headers = {'ContentType': 'application/json'}
        try:
            r = request.get_json()
            top_n = r.get('top_n')
            lang = r.get('lang')
            method = r.get('method', "weighted")
            clustered_docs = r.get('clustered_docs')
            if lang == 'en':
                stops = stopwords.words('english')
            if lang == 'de':
                stops = stopwords.words('german')
            else:
                stops = []

            try:
                tfidf_ranks = get_tfidfranks(clustered_docs, stops)
            except Exception as e:
                summarization_ns.logger.error(e)
            summaries = []
            for cluster, tfidf_scores in zip(clustered_docs, tfidf_ranks):
                try:
                    # get nc embeddings
                    assert isinstance(cluster, list), "cluster not a list"
                    payload = {}
                    payload["sents"] = cluster
                    embeddings = mnp.unpackb(requests.post(embed_url, json=payload).content)
                    textrank_scores = get_textrank(cluster, embeddings)
                except Exception as e:
                    summarization_ns.logger.error(e)
                    textrank_scores = [[1, token] for token in cluster]
                df1 = pd.DataFrame(textrank_scores, columns=['textrank', 'token'])
                df2 = pd.DataFrame(tfidf_scores, columns=['tfidf', 'token'])
                df = pd.merge(df1, df2, on='token')

                # implemented weighted and 2+1 methods here
                try:
                    summary = get_summary(df, method, weights=(0.5, 0.5), top_n=top_n)
                except Exception as e:
                    summarization_ns.logger.error(e)
                    summary = ""
                summaries.append(summary)
            result["summaries"] = summaries
            result["success"] = True
            return make_response(jsonify(result),
                                 200,
                                 headers)
        except Exception as e:
            summarization_ns.logger.error(e)
            result = {'success': False, 'reason': e}
            return make_response(jsonify(result),
                                 500,
                                 headers)


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
    df = df.groupby('token').mean().reset_index()
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
    # summarise with textrank
    sim_mat = cosine_similarity(normalize(embeddings))
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


def get_tfidfranks(docs, stops):
    """
    needs to be a list of lists of strings
    """
    assert isinstance(docs, list), "cluster not a list"
    for d in docs:
        assert isinstance(d, list), "doc not a list"
    docs = [" ".join([t.replace(" ", "_") for t in d])
            for d in docs]
    cv = CountVectorizer(lowercase=False, stop_words=stops)
    word_count_vector = cv.fit_transform(docs)
    token_names = cv.get_feature_names()
    token_names = [t.replace("_", " ") for t in token_names]
    tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_matrix = tfidf.fit(word_count_vector)
    rankings = []
    for doc in docs:
        tfidf_vector=tfidf.transform(cv.transform([doc])).tocoo()
        tuples = zip(tfidf_vector.col, tfidf_vector.data)
        scores = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        ranking = [(score, token_names[ind]) for ind, score in scores]
        rankings.append(ranking)
    return rankings
