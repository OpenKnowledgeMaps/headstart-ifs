import time
import json
import spacy
import redis
from spacy_cld import LanguageDetector


redis_store = redis.StrictRedis(host="localhost", port=6379, db=0)


class SpacyLangDetector(object):

    def __init__(self):
        self.langdetect = spacy.load('en_core_web_sm',
                                     disable=['tagger', 'parser', 'ner'])
        self.langdetect.add_pipe(LanguageDetector())

    def detect_langs_batch(self, docs):
        langs = [d._.languages[0] if len(d._.languages) > 0 else "unknown"
                 for d in self.langdetect.pipe(docs, batch_size=100, n_threads=5)]
        return langs


def run_lang_detect_process_batch():
    detector = SpacyLangDetector()
    while True:
        q = redis_store.lpop('batch_lang_detect')
        if q is not None:
            d = json.loads(q.decode('utf-8'))
            k = d["id"]
            if not type(d["docs"]) == list:
                docs = json.loads(d["docs"])
            else:
                docs = d["docs"]
            langs = detector.detect_langs_batch(docs)
            result = {"detected_langs": langs}
            redis_store.set(k, json.dumps(result))
        else:
            time.sleep(0.5)
