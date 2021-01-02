import sys
import logging
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from elasticsearch_dsl import Search, Q
import requests
import msgpack_numpy as mnp
from config import settings
from collections import Counter

mnp.patch()
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

lang_detect_url = "http://localhost/ifs/enrich/lang_detect"
batch_lang_detect_url = "http://localhost/ifs/enrich/lang_detect/batch"
sentenize_url = "http://localhost/ifs/enrich/sent_tokenize"
embed_url = "http://localhost/ifs/enrich/sent_embed/gusem"

class VectorSearch(object):

    def __init__(self, redis_store, settings, loglevel):
        self.settings = settings
        self.redis_store = redis_store
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(loglevel)
        self.logger.addHandler(handler)
        self.es = Elasticsearch(["127.0.0.1:9200"], timeout=60)

    def embed_query(self, query):
        payload = {}
        payload["doc"] = query
        lang = requests.post(lang_detect_url, json=payload).json()["detected_langs"]
        payload = {}
        payload["docs"] = [query]
        payload["lang"] = lang
        sents = requests.post(sentenize_url, json=payload).json()["sents"]
        payload = {}
        payload["sents"] = sents
        embeddings = mnp.unpackb(requests.post(embed_url, json=payload).content)
        embedded_query = embeddings.mean(0)
        return embedded_query

    def search_exact(self, query, n_results):
        embedded_query = self.embed_query(query)
        body = {
            "query": {
                "elastiknn_nearest_neighbors": {
                    "field": "knn_vec",
                    "vec": embedded_query.tolist(),
                    "model": "exact",
                    "similarity": "angular"
                }
            }
        }
        result = self.es.search(
                    index=self.settings.ES_INDEX,
                    body=body,
                    size=n_results)
        return result["hits"]["hits"]

    def search_ann(self, query, n_results):
        embedded_query = self.embed_query(query)
        body = {
            "query": {
                "elastiknn_nearest_neighbors": {
                    "field": "knn_vec",
                    "vec": {
                        "values": embedded_query.tolist()
                    },
                    "model": "lsh",
                    "similarity": "angular",
                    "candidates": n_results
                }
            }
        }
        result = self.es.search(
                    index=self.settings.ES_INDEX,
                    body=body,
                    size=n_results)
        return result["hits"]["hits"]

    def search_index(self, query, n_results):
        embedded_query = self.embed_query(query)
        body = {
            "query": {
                "elastiknn_nearest_neighbors": {
                    "field": "knn_vec",
                    "vec": {
                        "values": embedded_query.tolist()
                    },
                    "model": "lsh",
                    "similarity": "angular",
                    "candidates": n_results
                }
            }
        }
        result = self.es.search(
                    index=self.settings.ES_INDEX,
                    body=body,
                    size=n_results)
        return result["hits"]["hits"]

    def search_combined(self, query, n_results):
        embedded_query = self.embed_query(query)
        body = {
            "query": {
                "function_score": {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title", "description", "fulltext"]
                        }
                    },
                    "boost_mode": "replace",
                    "functions": [
                        {
                            "elastiknn_nearest_neighbors": {
                                "field": "knn_vec",
                                "vec": {
                                    "values": embedded_query.tolist()
                                },
                                "model": "lsh",
                                "similarity": "angular",
                                "candidates": n_results
                            }
                        }
                    ]
                }
            }
        }
        result = self.es.search(
                    index=self.settings.ES_INDEX,
                    body=body,
                    size=n_results)
        return result["hits"]["hits"]