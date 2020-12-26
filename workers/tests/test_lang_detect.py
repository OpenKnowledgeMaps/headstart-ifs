import json
import requests
from elasticsearch import Elasticsearch


def test_lang_detect():
    es = Elasticsearch()
    res = es.search(index="econ",
                    body = {
                            'query': {
                                'match_all' : {}
                            }
                    }
            )
    data = [doc for doc in res['hits']['hits']]
    url = "http://localhost/ifs/enrich/lang_detect"
    for d in data:
        payload = {}
        payload["doc"] = d["_source"]["fulltext"]
        res = requests.post(url,json=payload)
        result = res.json()
        assert result["success"] is True


def test_lang_detect_batch():
    es = Elasticsearch()
    res = es.search(index="econ",
                    body = {
                            'query': {
                                'match_all' : {}
                            }
                    }
            )
    data = [doc for doc in res['hits']['hits']]
    docs = [d["_source"]["fulltext"] for d in data]
    url = "http://localhost/ifs/enrich/lang_detect/batch"
    payload = {}
    payload["docs"] = docs
    res = requests.post(url,json=payload)
    result = res.json()
    assert result["success"] is True