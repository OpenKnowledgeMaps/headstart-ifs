import sys
import logging
import json
import uuid
import re
import requests
import numpy as np
import msgpack
import msgpack_numpy as mnp
from dateutil import parser
from datetime import datetime
from hashlib import md5
from tqdm import tqdm
from itertools import chain

mnp.patch()

import pymongo
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk
from elasticsearch_dsl import Search, Q, connections, Index
from models import Item

from tqdm import tqdm
from config import settings


formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

lang_detect_url = "http://localhost/ifs/enrich/lang_detect"
batch_lang_detect_url = "http://localhost/ifs/enrich/lang_detect/batch"
sentenize_url = "http://localhost/ifs/enrich/sent_tokenize"
embed_url = "http://localhost/ifs/enrich/sent_embed/gusem"

connections.configure(
    default={'hosts': '127.0.0.1:9200'},
    dev={
        'hosts': ['esdev1.example.com:9200'],
        'sniff_on_start': True
    }
)

def extract_date(url):
    datestring = "-".join(re.findall(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', url)[0])
    return datetime.strptime(datestring, '%Y-%m-%d')


def extract_categories(url):
    return re.findall(r'economist.com\/(\w+)', url)


def preprocess(doc):
    res = {}
    res["description"] = " ".join(doc.get('description'))
    res["fulltext"] = " ".join(doc.get('fulltext'))
    res["title"] = doc.get('title')[0]
    res["url"] = doc.get('url')[0]
    res["publication_date"] = extract_date(doc.get('url')[0])
    res["categories"] = extract_categories(doc.get('url')[0])
    return res

def sanity_check(doc):
    check = True
    if len(doc["fulltext"][0]) == 0:
        check = False
    return check


class FaissIndexer(object):
    
    def __init__(self, settings, loglevel='INFO'):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(loglevel)
        self.logger.addHandler(handler)
        self.es = Elasticsearch()

    def init_index(self, dims):
        index = faiss.IndexFlatL2(dims)
        index = faiss.IndexIDMap(index)
        self.index = index

    def prepare_content(self):
        q = Q("exists", field="faiss_id") & Q("exists", field="embeddings_doc")
        s = Search(using=self.es, index=self.settings.ES_INDEX).query(q)
        res = s.scan()
        content = [(r["faiss_id"],
                   r["embeddings_doc"])
                  for r in res]
        # exclude indexed IDs
        return content
    
    def update_index(self):
        content = self.prepare_content()
        embeddings = np.vstack([r[1] for r in content]).astype(np.float32)
        ids = np.array([r[0] for r in content]).astype(np.int64)
        self.add_to_index(embeddings, ids)

    def add_to_index(self, embeddings, ids):
        embeddings = embeddings.astype(np.float32)
        ids = ids.astype(np.int64)
        self.index.add_with_ids(embeddings, ids)
    
    def bulk_index(self, producer):
        docs = list(producer)
        temp_ids = [d[0] for d in docs]
        embeddings_sents = [d[1] for d in docs]    
        faiss_ids = []
        for i, emb in enumerate(embeddings_sents):
            faiss_ids.extend([temp_ids[i]] * emb.shape[0])
        embeddings = np.vstack(embeddings_sents).astype(np.float32)
        ids = np.array(faiss_ids).astype(np.int64)
        self.index.add_with_ids(embeddings, ids)

    def write_index(self, file):
        faiss.write_index(self.index, file)


def sanitize_text(text):
    if text.isprintable():
        return text
    else:
        return text.encode('ascii', "ignore").decode('utf-8')


class ESIndexBuilder(object):
    
    def __init__(self, settings, loglevel='INFO'):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(loglevel)
        self.logger.addHandler(handler)
        self.es = Elasticsearch()

    def clean_setup_index(self):
        self.es.indices.delete(index=self.settings.ES_INDEX, ignore=[400, 404])
        self.index = Index(self.settings.ES_INDEX)
        self.index.settings(
            number_of_shards=1,
            number_of_replicas=0,
            elastiknn=True
        )
        self.index.delete(ignore=404)
        self.index.create()
        Item.init(index=self.settings.ES_INDEX, using=self.es)
        new_mapping = {
            "properties": {
                "knn_vec": {
                    "type": "elastiknn_dense_float_vector", # 1
                    "elastiknn": {
                        "dims": 512,                        # 2
                        "model": "lsh",                     # 3
                        "similarity": "angular",            # 4
                        "L": 99,                            # 5
                        "k": 1                              # 6
                    }
                }
            }
        }
        self.es.indices.put_mapping(new_mapping, index=self.settings.ES_INDEX)

    def add_mongo_source(self, settings):
        self.client = pymongo.MongoClient(host=settings.MONGODB_SERVER,
                                          port=settings.MONGODB_PORT,
                                          username=settings.MONGODB_USER,
                                          password=settings.MONGODB_PASSWORD,
                                          authSource=settings.MONGODB_AUTHSRC)
        self.mongo_db = self.client[settings.MONGODB_DB]
        self.mongo_collection = self.mongo_db[settings.MONGODB_COLLECTION]

    def url_exists(self, url):
        q = Q("match", url=url)
        s = Search(using=self.es, index=self.settings.ES_INDEX).query(q)
        return True if s.count() > 0 else False

    def update_batch(self, batch):
        ids = [r.meta.id for r in batch]
        batch = [r.to_dict() for r in batch]
        for i, id in enumerate(ids):
            batch[i]["id"] = id
        batch = prepare_batch(batch)                    
        for b in batch:
            update = {}
            # cat = b.get('category', '')
            # update["categories"] = cat if isinstance(cat, list) else [cat]
            update["lang"] = b.get('lang')
            update["embeddings_doc"] = b.get('embeddings_doc')
            doc = Item.get(b["id"], self.es, self.settings.ES_INDEX)
            doc.update(using=self.es, index=self.settings.ES_INDEX, **update)

    def add_missing_embeddings(self, batch_size):
        q = ~Q("exists", field="embeddings_doc")
        s = Search(using=self.es, index=self.settings.ES_INDEX).query(q)
        n_missing = len(list(s.scan()))
        print("Missing embeddings_doc: %d" %n_missing)
        res = s.params(scroll="20m").scan()
        batch = []
        for r in tqdm(res, total=n_missing):
            batch.append(r)
            if len(batch) >= batch_size:
                try:
                    self.update_batch(batch)
                    batch = []
                except Exception as e:
                    print(e)
                    batch = []
        if len(batch) > 0:
            try:
                self.update_batch(batch)
                batch = []
            except Exception as e:
                print(e)
                print(batch)
                batch = []

    def add_missing_faiss_ids(self):
        q = Q("exists", field="faiss_id")
        s = Search(using=self.es, index=self.settings.ES_INDEX).query(q).sort('-faiss_id')
        try:
            next_faiss_id = s.execute().hits.hits[0]["_source"]["faiss_id"] + 1
        except IndexError:
            next_faiss_id = 0
        q = ~Q("exists", field="faiss_id")
        s = Search(using=self.es, index=self.settings.ES_INDEX).query(q)
        n_missing = len(list(s.scan()))
        print("Missing faiss_ids: %d" %n_missing)
        res = s.scan()
        for r in tqdm(res, total=n_missing):
            update = {"faiss_id": next_faiss_id}
            doc = Item.get(r.meta.id, self.es, self.settings.ES_INDEX)
            doc.update(using=self.es, index=self.settings.ES_INDEX, **update)
            next_faiss_id += 1

def enrich(text):
    payload = {}
    payload["doc"] = text
    langs = requests.post(lang_detect_url, json=payload).json()["detected_langs"]
    lang = langs[0] if isinstance(langs, list) else langs
    payload = {}
    payload["docs"] = [text]
    payload["lang"] = lang
    sents = requests.post(sentenize_url, json=payload).json()["sents"]
    payload = {}
    payload["sents"] = sents
    embeddings = mnp.unpackb(requests.post(embed_url, json=payload).content)
    return lang, embeddings


def prepare_batch(batch):
    payload = {}
    payload["docs"] = [sanitize_text(b.get('fulltext', '')) for b in batch]
    langs = requests.post(batch_lang_detect_url, json=payload).json()["detected_langs"]
    for i, lang in enumerate(langs):
        batch[i]["lang"] = lang
    lang_batches = {}
    for lang in set(langs):
        lang_batches[lang] = [b for b in batch if b.get('lang') == lang]

    for lang, lang_batch in lang_batches.items():
        payload = {}
        payload["docs"] = [sanitize_text(b.get('fulltext', '')) for b in lang_batch]
        payload["lang"] = lang
        sents = requests.post(sentenize_url, json=payload).json()["sents"]
        for i, s in enumerate(sents):
            lang_batches[lang][i]["sents"] = s
        n_sents = [len(s) for s in sents]
        payload = {}
        payload["sents"] = list(chain.from_iterable(sents))
        embeddings = mnp.unpackb(requests.post(embed_url, json=payload).content)
        splits = np.cumsum(n_sents)[:-1]
        embeddings = np.split(embeddings, splits)
        for i, emb in enumerate(embeddings):
            lang_batches[lang][i]["embeddings_doc"] =  emb.mean(0).tolist()
    return list(chain.from_iterable(lang_batches.values()))


def docs_producer(batch):
    for b in batch:
        d = {k:v for k,v in b.items() if not k in ["sents", "embeddings_sents"]}
        d = Item(**d)
        yield d.to_dict(include_meta=True)


def ingest_batch(esi, batch):
    batch = prepare_batch(batch)
    bulk(esi.es, docs_producer(batch, "es"), index=esi.settings.ES_INDEX)

def main(batch_size=100, tabular_rasa=False):
    esi = ESIndexBuilder(settings)
    esi.add_mongo_source(settings)
    n_total = esi.mongo_collection.count()
    print(n_total)
    if tabular_rasa:
        esi.clean_setup_index()


    q = Q("exists", field="url")
    s = Search(using=esi.es, index=settings.ES_INDEX).query(q)
    existing_urls = set((r.url for r in s.scan()))
    batch = []
    for doc in tqdm(esi.mongo_collection.find(), total=n_total):
        if sanity_check(doc):
            try:
                doc = preprocess(doc)
                if not tabular_rasa and doc["url"] in existing_urls:
                    continue
                batch.append(doc)
            except Exception as e:
                print(e)

            if len(batch) >= batch_size:
                try:
                    ingest_batch(esi, batch)
                    batch = []
                except Exception as e:
                    print(e)
                    batch = []
    # process incomplete batch
    if len(batch) > 0:
        try:
            ingest_batch(esi, batch)
        except Exception as e:
            print(e)

    print(esi.es.count())

if __name__ == '__main__':
    main(50, True)