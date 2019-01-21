import time
import json
import redis
import os
from gensim.models import Doc2Vec

redis_store = redis.StrictRedis(host="localhost", port=6379, db=0)


class Embedder(object):

    def __init__(self, model_path, model_name):
        """model_path: headstart-models/models"""
        path = os.path.join(model_path, model_name)
        self.model = Doc2Vec.load(path)

    def get_docvec(self, doc):
        """doc should be a list of tokens"""
        return self.model.infer_vector(doc, epochs=30)

    def get_docvecs_batch(self, docs):
        return [self.get_docvec(doc) for doc in docs]

    def embed_noun_chunks(self, doc):
        """
        doc is a list of noun_chunks
        """
        return [self.get_docvecs_batch(nc.lower().split()) for nc in doc]

    def embed_noun_chunks_batch(self, docs):
        """docs is a list of docs which are lists of noun_chunks"""
        return [self.embed_noun_chunks(doc)
                for doc in docs]


def run_embed_process_batch():
    embedder = Embedder(model_path="/home/chris/projects/OpenKnowledgeMaps/headstart-models/models",
                        model_name="core2vec_model_en")
    while True:
        q = redis_store.lpop('batch_embed')
        if q is not None:
            d = json.loads(q.decode('utf-8'))
            k = d["id"]
            if not type(d["docs"]) == list:
                docs = json.loads(d["docs"])
            else:
                docs = d["docs"]
            docvecs = embedder.get_docvecs_batch(docs)
            result = {"docvecs": docvecs}
            redis_store.set(k, json.dumps(result))
        else:
            time.sleep(0.5)


def run_embed_noun_chunks_process():
    embedder = Embedder(model_path="/home/chris/projects/OpenKnowledgeMaps/headstart-models/models",
                        model_name="core2vec_model_en")
    while True:
        q = redis_store.lpop('embed_noun_chunks')
        if q is not None:
            d = json.loads(q.decode('utf-8'))
            k = d["id"]
            if not type(d["doc"]) == list:
                doc = json.loads(d["doc"])
            else:
                doc = d["doc"]
            docvecs = embedder.embed_noun_chunks(doc)
            result = {"embeddings": docvecs}
            redis_store.set(k, json.dumps(result))
        else:
            time.sleep(0.5)


def run_embed_noun_chunks_process_batch():
    embedder = Embedder(model_path="/home/chris/projects/OpenKnowledgeMaps/headstart-models/models",
                        model_name="core2vec_model_en")
    while True:
        q = redis_store.lpop('batch_embed_noun_chunks')
        if q is not None:
            d = json.loads(q.decode('utf-8'))
            k = d["id"]
            if not type(d["docs"]) == list:
                docs = json.loads(d["docs"])
            else:
                docs = d["docs"]
            docvecs = embedder.embed_noun_chunks_batch(docs)
            result = {"embeddings": docvecs}
            redis_store.set(k, json.dumps(result))
        else:
            time.sleep(0.5)
