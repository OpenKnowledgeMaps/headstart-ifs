import time
import uuid
import json
from flask import Blueprint, request, jsonify
import redis
import numpy as np
import igraph as ig
from sklearn.metrics.pairwise import cosine_similarity

redis_store = redis.StrictRedis(host="localhost", port=6379, db=0)


app = Blueprint('tagger', __name__)


@app.route('/summarize_clusters', methods=['GET', 'POST'])
def summarize_clusters():
    # docs, cluster_ids
    pass


@app.route('/summarize', methods=['GET', 'POST'])
def summarize_doc():
    # doc
    """
    if method is noun_chunks, doc should be a list of noun chunks
    """
    response = {"success": False}
    if request.method == 'POST':
        r = request.get_json()
        method = r.get('method')
        top_n = r.get('top_n')

        if method == 'noun_chunks':
            noun_chunks = json.loads(r.get('doc'))
            # get nc embeddings
            k = str(uuid.uuid4())
            d = {"id": k, "doc": noun_chunks}
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
                ranked_nc = get_ranking(noun_chunks, embeddings)
                summary = []
                for rnc in ranked_nc:
                    candidate = rnc[1]
                    if candidate not in summary:
                        if len(candidate) < 35:
                            summary.append(candidate.replace(" - ", "-"))
                response["summary"] = ", ".join(summary[:top_n])
                response["success"] = True
            except Exception as e:
                response["msg"] = str(e)
                response["summary"] = None
        else:
            response["msg"] = "Method not implemented, choose one of ['noun_chunks']"

    return jsonify(response)


def get_ranking(tokens, embeddings):
    res = [(x, np.array(y))
           for x, y in zip(tokens, embeddings)
           if y is not None]
    # average over noun chunk word vectors for each noun_chunk
    noun_chunks = [r[0] for r in res]
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
                      for i, nc in enumerate(noun_chunks)), reverse=True)
    return ranking
