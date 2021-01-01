import sys
import logging
import numpy as np
import faiss
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from elasticsearch_dsl import Search, Q
import requests
import msgpack_numpy as mnp
from config import settings

mnp.patch()
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

lang_detect_url = "http://localhost/ifs/enrich/lang_detect"
batch_lang_detect_url = "http://localhost/ifs/enrich/lang_detect/batch"
sentenize_url = "http://localhost/ifs/enrich/sent_tokenize"
embed_url = "http://localhost/ifs/enrich/sent_embed"

class FSearch(object):

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

    def load_index(self, path):
        self.index = faiss.read_index(path)

    def embed_query(self, query):
        payload = {}
        payload["doc"] = query
        lang = requests.post(lang_detect_url, json=payload).json()["detected_langs"]
        payload = {}
        payload["docs"] = [query]
        payload["lang"] = lang
        sents = requests.post(sentenize_url, json=payload).json()["sents"]
        payload = {}
        payload["doc"] = "\n".join(["\n".join(s) for s in sents])
        payload["lang"] = lang
        embeddings = mnp.unpackb(requests.post(embed_url, json=payload).content)
        embedded_query = embeddings.mean(0)
        return embedded_query

    def search(self, embedded_query, n_results):
        D, I = self.index.search(np.array([embedded_query]), k=n_results)
        result_ids = I.tolist()[0]
        return result_ids

    def retrieve_results(self, result_ids, n_results):
        s = Search(using=self.es, index=self.settings.ES_INDEX)
        s = s.filter('terms', faiss_id=result_ids)[:n_results]
        result_docs = s.execute()
        return result_docs


