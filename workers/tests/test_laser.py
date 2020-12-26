import json
import requests
from elasticsearch import Elasticsearch


def test_sent_embed():
    es = Elasticsearch()
    res = es.search(index="econ",
                    body = {
                            'query': {
                                'match_all' : {}
                            }
                    }
            )
    data = [doc for doc in res['hits']['hits']]
    url = "http://localhost/ifs/enrich/sent_embed"
    for d in data:
        payload = {}
        payload["doc"] = d["_source"]["fulltext"]
        payload["lang"] = "en"
        res = requests.post(url,json=payload)
        result = res.json()
        assert result["success"] is True