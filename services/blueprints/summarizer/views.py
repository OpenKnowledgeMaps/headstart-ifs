import time
import uuid
import json
from flask import Blueprint, request, jsonify
import redis
import numpy as np
import networkx as nx
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
            res = [(x, np.array(y))
                   for x, y in zip(noun_chunks, embeddings)
                   if y is not None]
            # average over noun chunk word vectors for each noun_chunk
            noun_chunks = [r[0] for r in res]
            embeddings = [r[1] for r in res]
            # summarise with textrank
            sim_mat = cosine_similarity(embeddings)
            np.fill_diagonal(sim_mat, 0)
            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)
            ranked_nc = sorted(((scores[i], nc)
                                for i, nc in enumerate(noun_chunks)), reverse=True)
            response["summary"] = ", ".join([rnc[1]
                                            for rnc in ranked_nc[:top_n]])
            response["success"] = True
        else:
            response["msg"] = "Method not implemented, choose one of ['noun_chunks']"

    return jsonify(response)
