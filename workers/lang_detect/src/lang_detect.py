import sys
import logging
import time
import json
import spacy
from spacy_cld import LanguageDetector


formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

                              
class SpacyLangDetector(object):

    def __init__(self, redis_store, loglevel):
        self.redis_store = redis_store
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(loglevel)
        self.logger.addHandler(handler)
        self.langdetect = spacy.load('en_core_web_sm',
                                     disable=['tagger', 'parser', 'ner'])
        self.langdetect.add_pipe(LanguageDetector())

    def next_item(self):
        queue, msg = self.redis_store.blpop("batch_lang_detect")
        msg = json.loads(msg)
        k = msg.get('id')
        docs = msg.get('docs')
        return k, docs

    def detect_langs_batch(self, docs):
        langs = [d._.languages[0] if len(d._.languages) > 0 else "unknown"
                 for d in self.langdetect.pipe(docs, batch_size=100, n_threads=5)]
        return langs

    def run(self):
        while True:
            k, docs = self.next_item()
            try:
                res = {}
                res["id"] = k
                res["langs"] = self.detect_langs_batch(docs)
                self.redis_store.set(k, json.dumps(res))
            except Exception as e:
                self.logger.error(e)